import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SDLoRADehaze(nn.Module):
    """Single-step diffusion dehazing: frozen SD2.1-base + trainable UNet LoRA.

    The hazy-image latent is used directly as the diffusion starting point
    (no random noise is added). The UNet predicts the noise residual at a
    fixed timestep, and the clean latent is reconstructed analytically in one
    step, so the output is fully deterministic. Only the injected LoRA layers
    are trainable; the VAE and the UNet backbone stay frozen.

    The forward pass is a pixel-to-pixel mapping: input/output are RGB image
    tensors in [0, 1] of shape (B, 3, H, W), which makes this arch a drop-in
    network for SRModel-style training. H and W must be multiples of 8 (the
    VAE scale factor); SDLoRAModel.test() reflect-pads arbitrary sizes.

    Args:
        sd_path (str): Local path of the SD2.1-base snapshot (with vae/,
            unet/, scheduler/, text_encoder/, tokenizer/ subfolders).
        lora_rank (int): LoRA rank. Default: 8.
        lora_alpha (float): LoRA alpha. Default: same as lora_rank.
        lora_dropout (float): LoRA dropout. Default: 0.
        lora_targets (list[str]): UNet submodules to inject LoRA into.
            Default: attention projections ('to_q', 'to_k', 'to_v', 'to_out.0').
        lora_conv_targets (list[str]): Additional conv-layer LoRA targets for
            a larger trainable budget (e.g. ['conv1', 'conv2', 'conv_shortcut',
            'conv_in', 'conv_out']). Default: empty (attention-only).
        fixed_timestep (int): The fixed diffusion timestep. Default: 999.
        prediction (str): 'eps' — UNet predicts the noise residual and the
            clean latent is reconstructed analytically (OSEDiff-style);
            'x0' — the UNet directly predicts the clean latent (faster to
            fit, loses the diffusion-ODE interpretation). Default: 'eps'.
        vae_fp32 (bool): Run the VAE in fp32 (numerical-safety fallback).
            Default: False (fp16).
        gradient_checkpointing (bool): Enable UNet gradient checkpointing to
            save training VRAM. Default: True.
    """

    def __init__(self,
                 sd_path='experiments/pretrained_models/sd21base',
                 lora_rank=8,
                 lora_alpha=None,
                 lora_dropout=0.0,
                 lora_targets=('to_q', 'to_k', 'to_v', 'to_out.0'),
                 lora_conv_targets=(),
                 fixed_timestep=999,
                 prediction='eps',
                 vae_fp32=False,
                 gradient_checkpointing=True):
        super().__init__()
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig

        if lora_alpha is None:
            lora_alpha = lora_rank
        assert prediction in ('eps', 'x0')
        self.prediction = prediction
        self.scale_factor = 0.18215
        self.fixed_timestep = fixed_timestep
        self.vae_fp32 = vae_fp32

        scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder='scheduler')
        alpha_bar = scheduler.alphas_cumprod[fixed_timestep].reshape(1).float()
        self.register_buffer('alpha_bar', alpha_bar)
        del scheduler

        vae_dtype = torch.float32 if vae_fp32 else torch.float16
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae', torch_dtype=vae_dtype)
        self.vae.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet', torch_dtype=torch.float16)
        self.unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_targets) + list(lora_conv_targets))
        self.unet.add_adapter(lora_config)
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # precompute the empty-prompt embedding, then drop the text encoder
        null_emb = self._encode_null_prompt(sd_path)
        self.register_buffer('null_emb', null_emb)

        # keep the small LoRA params in fp32 for stable optimization and EMA
        for name, param in self.unet.named_parameters():
            if 'lora_' in name:
                param.data = param.data.float()

    @staticmethod
    def _encode_null_prompt(sd_path):
        from transformers import CLIPTextModel, CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer')
        # fp32 on CPU: fp16 LayerNorm is not implemented on CPU
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder='text_encoder', torch_dtype=torch.float32)
        tokens = tokenizer(
            [''], padding='max_length', max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        with torch.no_grad():
            emb = text_encoder(tokens.input_ids)[0]  # (1, 77, 1024)
        del text_encoder, tokenizer
        return emb.half()

    def forward(self, x):
        """x: (B, 3, H, W) float in [0, 1]. Returns the dehazed image (fp32)."""
        vae_dtype = torch.float32 if self.vae_fp32 else torch.float16
        # hazy latent as the diffusion starting point (no noise added)
        with torch.no_grad():
            z = self.vae.encode((x * 2 - 1).to(vae_dtype)).latent_dist.mean
        z = (z * self.scale_factor).to(torch.float16)

        b = z.shape[0]
        t = torch.full((b, ), self.fixed_timestep, device=z.device, dtype=torch.long)
        noise_pred = self.unet(z, t, encoder_hidden_states=self.null_emb.expand(b, -1, -1)).sample

        if self.prediction == 'x0':
            # the UNet directly predicts the clean latent
            pred_x0 = noise_pred
        else:
            # epsilon-prediction: analytic one-step x0 reconstruction
            pred_x0 = (z - (1 - self.alpha_bar).sqrt() * noise_pred) / self.alpha_bar.sqrt()

        img = self.vae.decode((pred_x0 / self.scale_factor).to(vae_dtype)).sample
        img = (img.float() + 1) / 2
        if not self.training:
            img = img.clamp(0, 1)
        return img
