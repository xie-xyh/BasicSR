import torch

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class DEANetV8Model(DehazeModel):
    """DehazeModel variant that logs DEANetV8's internal diagnostics.

    Forwards the arch's per-batch diagnostics (predicted boundary fc, veil
    magnitude, skip-gate suppression strength) to the training log so the
    012b probe can watch for boundary collapse and dead gates (see
    research/012_deanetv8/README.md 8k diagnostic checklist).
    """

    def optimize_parameters(self, current_iter):
        super().optimize_parameters(current_iter)
        diagnostics = self.get_bare_model(self.net_g).diagnostics
        for key, value in diagnostics.items():
            self.log_dict[key] = value.item() if torch.is_tensor(value) else value
