"""The final EGMF-NET: RGB, MSCAN, EMCAD-Lite, gated fusion, configurable cascade."""
import torch
from torch import nn
import torch.nn.functional as F

from .blocks import SharedEncoder, EMCADLiteBlock, GuidedEMCADLiteBlock, dice_loss_from_logits


class Branch(nn.Module):
    """One encoder with enhancement and enhancement-guided binary decoders."""
    def __init__(self, base_channels=32, channels=(64, 128, 256, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder = SharedEncoder(3, base_channels, channels)
        widths = [(c4, c3, c3), (c3, c2, c2), (c2, c1, c1),
                  (c1, base_channels, base_channels), (base_channels, base_channels, base_channels)]
        for index, (input_dim, skip_dim, output_dim) in enumerate(widths, 1):
            setattr(self, f"enh_up{index}", EMCADLiteBlock(input_dim, skip_dim, output_dim))
            setattr(self, f"bin_up{index}", GuidedEMCADLiteBlock(input_dim, skip_dim, output_dim, output_dim))
        self.residual_head = nn.Conv2d(base_channels, 3, 1)
        self.logits = nn.Conv2d(base_channels, 1, 1)

    def forward(self, image):
        x1, x2, x3, x4, x5, bottleneck = self.encoder(image)
        enhanced_feature = binary_feature = bottleneck
        for index, skip in enumerate((x5, x4, x3, x2, x1), 1):
            enhanced_feature = getattr(self, f"enh_up{index}")(enhanced_feature, skip)
            binary_feature = getattr(self, f"bin_up{index}")(binary_feature, skip, enhanced_feature)
        residual = self.residual_head(enhanced_feature).tanh()
        enhanced = ((image + 1) * 0.5 + residual).clamp(0, 1)
        return {"bin_logits": self.logits(binary_feature), "enhanced_pred": enhanced,
                "pred_residual": residual, "decoder_feature": binary_feature}


def crop_full_to_patch(tensor, crop_meta, patch_shape):
    """Map original-page crop coordinates into the resized global feature map."""
    patch_h, patch_w = patch_shape
    height, width = tensor.shape[-2:]
    patches = []
    for index, meta in enumerate(crop_meta.detach().cpu().tolist()):
        left, top, crop_w, crop_h, original_w, original_h = meta
        x1 = max(0, min(round(left / original_w * width), width - 1))
        y1 = max(0, min(round(top / original_h * height), height - 1))
        x2 = max(x1 + 1, min(round((left + crop_w) / original_w * width), width))
        y2 = max(y1 + 1, min(round((top + crop_h) / original_h * height), height))
        region = tensor[index:index + 1, :, y1:y2, x1:x2]
        patches.append(F.interpolate(region, size=(patch_h, patch_w), mode="bilinear", align_corners=False))
    return torch.cat(patches)


class EGMFNet(nn.Module):
    """Global/local gated stages with shared or independent parameters."""

    def __init__(self, base_channels=32, channels=(64, 128, 256, 256), stages=2, independent_cascade_params=False):
        super().__init__()
        if type(stages) is not int or stages <= 0:
            raise ValueError("stages must be a positive integer")
        if type(independent_cascade_params) is not bool:
            raise ValueError("independent_cascade_params must be a boolean")
        self.stages = stages
        self.independent_cascade_params = independent_cascade_params
        self.cascade_stage_models = None
        if independent_cascade_params and stages > 1:
            self.cascade_stage_models = nn.ModuleList(
                EGMFNet(base_channels, channels, stages=1) for _ in range(stages)
            )
            return
        self.full_branch = Branch(base_channels, channels)
        self.patch_branch = Branch(base_channels, channels)
        self.bin_gate_head = nn.Conv2d(base_channels * 2, 2, 3, padding=1)
        self.enhance_gate_head = nn.Conv2d(base_channels * 2, 6, 3, padding=1)

    def get_stage_model(self, index):
        if not 0 <= index < self.stages:
            raise IndexError("cascade stage index out of range")
        return self if self.cascade_stage_models is None else self.cascade_stage_models[index]

    @staticmethod
    def _gated_mix(global_tensor, local_tensor, gate_logits):
        batch, channels, height, width = global_tensor.shape
        gates = gate_logits.view(batch, 2, channels, height, width).softmax(dim=1)
        return gates[:, 0] * global_tensor + gates[:, 1] * local_tensor

    def fuse(self, global_output, local_output):
        features = torch.cat([global_output["decoder_feature"], local_output["decoder_feature"]], dim=1)
        logits = self._gated_mix(global_output["bin_logits"], local_output["bin_logits"], self.bin_gate_head(features))
        enhanced = self._gated_mix(global_output["enhanced_pred"], local_output["enhanced_pred"], self.enhance_gate_head(features)).clamp(0, 1)
        return logits, enhanced

    def forward(self, full_image, patch_image, crop_meta):
        current_full, current_patch = full_image, patch_image
        outputs = []
        for stage in range(self.stages):
            stage_model = self.get_stage_model(stage)
            full = stage_model.full_branch(current_full)
            local = stage_model.patch_branch(current_patch)
            aligned = {key: crop_full_to_patch(value, crop_meta, patch_image.shape[-2:])
                       for key, value in full.items()}
            logits, enhanced = stage_model.fuse(aligned, local)
            outputs.append({"bin_logits": logits, "enhanced_pred": enhanced,
                            "full_logits": full["bin_logits"], "patch_logits": local["bin_logits"],
                            "full_enhanced": full["enhanced_pred"], "patch_enhanced": local["enhanced_pred"]})
            if stage + 1 < self.stages:
                current_full = full["enhanced_pred"] * 2 - 1
                current_patch = enhanced * 2 - 1
        return {**outputs[-1], "stage_outputs": outputs}


def compute_loss(output, target, enhance_target, full_target, full_enhance_target):
    """Weight all cascade stages equally."""
    def binary(logits, truth):
        return F.binary_cross_entropy_with_logits(logits.float(), truth.float()) + dice_loss_from_logits(logits, truth)

    binary_losses, enhancement_losses = [], []
    for stage in output["stage_outputs"]:
        binary_losses.append(0.5 * binary(stage["bin_logits"], target)
                             + 0.3 * binary(stage["full_logits"], full_target)
                             + 0.3 * binary(stage["patch_logits"], target))
        enhancement_losses.append(0.3 * F.l1_loss(stage["enhanced_pred"].float(), enhance_target.float())
                                 + 0.1 * F.l1_loss(stage["full_enhanced"].float(), full_enhance_target.float())
                                 + 0.1 * F.l1_loss(stage["patch_enhanced"].float(), enhance_target.float()))
    count = len(binary_losses)
    weights = [1.0 / count] * count
    loss_bin = sum(weight * loss for weight, loss in zip(weights, binary_losses))
    loss_enhance = sum(weight * loss for weight, loss in zip(weights, enhancement_losses))
    return {"loss_total": loss_bin + loss_enhance,
            "loss_bin": loss_bin.detach(), "loss_enhance": loss_enhance.detach()}
