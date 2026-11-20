import torch

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class WaveletCalibrationModel(DehazeModel):
    """DehazeModel variant for CalibratedWaveletUNet (011l).

    The backbone is frozen, so EMA only needs to track the color head; the
    backbone of net_g_ema stays at the deepcopy from initialization, which is
    identical to the frozen backbone. decay=0 copies the full state (used for
    checkpoint export).
    """

    @torch.no_grad()
    def model_ema(self, decay=0.999):
        source = self.get_bare_model(self.net_g)
        if decay == 0:
            self.net_g_ema.load_state_dict(source.state_dict())
            return
        for averaged, current in zip(self.net_g_ema.color_head.parameters(), source.color_head.parameters()):
            averaged.lerp_(current, 1 - decay)
