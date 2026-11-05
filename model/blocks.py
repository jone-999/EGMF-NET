"""Fixed RGB MSCAN / EMCAD-Lite building blocks for EGMF-NET."""
import torch
from torch import nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, target, smooth=1e-6):
    logits, target = logits.float(), target.float()
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()

def make_norm(channels, norm_type="group", groups=8):
    groups = max(1, min(int(groups), int(channels)))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type="group", groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            make_norm(out_channels, norm_type, groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="group", groups=8):
        super().__init__()
        self.block = nn.Sequential(
            ConvAct(in_channels, out_channels, 3, norm_type, groups),
            ConvAct(out_channels, out_channels, 3, norm_type, groups),
        )

    def forward(self, x):
        return self.block(x)

class MSCANBlock(nn.Module):
    """Small SegNeXt-style multi-scale convolutional attention block."""

    def __init__(self, channels, norm_type="group", groups=8):
        super().__init__()
        self.norm = make_norm(channels, norm_type, groups)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.strip_h = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False)
        self.strip_v = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False)
        self.mix = nn.Conv2d(channels, channels, 1, bias=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x):
        residual = x
        attn = self.norm(x)
        attn = self.local(attn) + self.strip_h(attn) + self.strip_v(attn)
        x = residual + self.mix(attn) * residual
        return x + self.ffn(self.norm(x))

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="group", groups=8, downsample=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()
        self.proj = ConvBlock(in_channels, out_channels, norm_type, groups)
        self.context = MSCANBlock(out_channels, norm_type, groups)

    def forward(self, x):
        x = self.proj(self.pool(x))
        return self.context(x)

class SharedEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, condition_dim_list, norm_type="group", groups=8):
        super().__init__()
        c1, c2, c3, c4 = [int(v) for v in condition_dim_list]
        self.stem = EncoderStage(in_channels, base_channels, norm_type, groups, downsample=False)
        self.down1 = EncoderStage(base_channels, base_channels, norm_type, groups)
        self.down2 = EncoderStage(base_channels, c1, norm_type, groups)
        self.down3 = EncoderStage(c1, c2, norm_type, groups)
        self.down4 = EncoderStage(c2, c3, norm_type, groups)
        self.down5 = EncoderStage(c3, c4, norm_type, groups)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        bottleneck = self.down5(x5)
        return x1, x2, x3, x4, x5, bottleneck

class EMCADLiteBlock(nn.Module):
    """Lightweight EMCAD-inspired up block with multi-scale DWConv and gates."""

    def __init__(self, in_channels, skip_channels, out_channels, norm_type="group", groups=8):
        super().__init__()
        self.dec_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, 1, bias=False)
        self.fuse = ConvAct(out_channels * 2, out_channels, 1, norm_type, groups)
        self.dw3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.dw5 = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels, bias=False)
        self.dw7 = nn.Conv2d(out_channels, out_channels, 7, padding=3, groups=out_channels, bias=False)
        self.ms_mix = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            make_norm(out_channels, norm_type, groups),
            nn.SiLU(inplace=True),
        )
        hidden = max(out_channels // 4, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(out_channels, 1, 7, padding=3), nn.Sigmoid())
        self.out = ConvAct(out_channels, out_channels, 3, norm_type, groups)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec_proj(x)
        skip = self.skip_proj(skip)
        fused = self.fuse(torch.cat([x, skip], dim=1))
        ms = self.ms_mix(torch.cat([self.dw3(fused), self.dw5(fused), self.dw7(fused)], dim=1))
        gated = ms * self.channel_gate(ms) * self.spatial_gate(ms)
        return self.out(fused + gated)

class GuidedEMCADLiteBlock(nn.Module):
    """Add enhancement decoder features to the binary decoder."""
    def __init__(self, in_channels, skip_channels, enhance_channels, out_channels, norm_type="group", groups=8):
        super().__init__()
        self.adapt = nn.Conv2d(enhance_channels, in_channels, 1, bias=True)
        self.block = EMCADLiteBlock(in_channels, skip_channels, out_channels, norm_type, groups)

    def forward(self, x, skip, enhance_feature):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        guidance = F.interpolate(enhance_feature, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(x + self.adapt(guidance), skip)

