"""
Prism v2 — U-Net + Transformer for photorealistic style transfer.

Architecture designed for <10ms inference at 1080p on RTX 5060 Ti:
  - Conv encoder downsamples to 1/8 resolution (tokenizer)
  - Transformer blocks at low resolution (global understanding)
  - Conv decoder upsamples with skip connections (spatial detail)
  - PixelShuffle to display resolution

Key insight from FSR 4: expensive layers run at LOW resolution.
Key insight from DLSS 5: transformers provide global context for style.

Still trained as GAN with PatchDiscriminator — same training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import math


# ============================================================================
# Config
# ============================================================================

@dataclass
class ModelConfig:
    # Encoder channels at each resolution level
    enc_channels: list = field(default_factory=lambda: [32, 64, 128])

    # Transformer at bottleneck
    bottleneck_ch: int = 128         # channels at lowest resolution
    n_transformer_blocks: int = 4    # transformer depth (global context)
    n_heads: int = 4                 # attention heads
    ffn_ratio: int = 4               # FFN expansion ratio
    window_size: int = 8             # 0 = full attention, >0 = windowed

    # Decoder
    dec_channels: list = field(default_factory=lambda: [128, 64, 32])

    # Temporal
    temporal: str = "gru"            # "none", "ema", "gru"

    # Input
    use_warped_prev: bool = True
    input_channels: int = 0          # auto: 6 + 3 = 9

    # Output
    scale: int = 2                   # PixelShuffle scale

    def __post_init__(self):
        self.input_channels = 6 + (3 if self.use_warped_prev else 0)


# ============================================================================
# Building blocks
# ============================================================================

class DSCBlock(nn.Module):
    """Lightweight depthwise separable residual block."""
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return F.relu(self.pw(self.dw(x)) + x, inplace=True)


class DownBlock(nn.Module):
    """Strided conv downsample + residual block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.block = DSCBlock(out_ch)

    def forward(self, x):
        return self.block(F.relu(self.down(x), inplace=True))


class UpBlock(nn.Module):
    """Transpose conv upsample + skip connection + residual block."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 1)
        self.block = DSCBlock(out_ch)

    def forward(self, x, skip):
        x = F.relu(self.up(x), inplace=True)
        # Resize if needed (in case of odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(F.relu(self.fuse(x), inplace=True))


# ============================================================================
# Transformer blocks (global context at low resolution)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention. At 67×120 = 8K tokens this is fast."""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention (PyTorch 2.x — fused, memory-efficient)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> Attention -> LayerNorm -> FFN."""
    def __init__(self, dim, n_heads=4, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(dim * ffn_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Motion vector warping (same as v1)
# ============================================================================

def warp(x, mv):
    B, _, H, W = x.shape
    if mv.shape[2:] != (H, W):
        scale_x = W / mv.shape[3]
        scale_y = H / mv.shape[2]
        mv = F.interpolate(mv, (H, W), mode="bilinear", align_corners=False)
        mv = mv * torch.tensor([scale_x, scale_y], device=mv.device).view(1, 2, 1, 1)

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    mv_norm = mv.permute(0, 2, 3, 1).to(x.dtype).clone()
    mv_norm[..., 0] /= W / 2
    mv_norm[..., 1] /= H / 2
    return F.grid_sample(x.float(), (grid - mv_norm).float(), mode="bilinear",
                         padding_mode="border", align_corners=True).to(x.dtype)


# ============================================================================
# Temporal (at bottleneck resolution — very cheap)
# ============================================================================

class TemporalGRU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Conv2d(ch * 2, ch * 2, 3, padding=1)
        self.cand = nn.Conv2d(ch * 2, ch, 3, padding=1)

    def forward(self, x, prev_hidden, mv):
        if prev_hidden is None:
            return x
        if mv is not None:
            prev_hidden = warp(prev_hidden, mv)
        combined = torch.cat([x, prev_hidden], dim=1)
        gates = torch.sigmoid(self.gate(combined))
        reset, update = gates.chunk(2, dim=1)
        cand = torch.tanh(self.cand(torch.cat([x, reset * prev_hidden], dim=1)))
        return update * prev_hidden + (1 - update) * cand


# ============================================================================
# Generator — U-Net + Transformer
# ============================================================================

class PrismV2(nn.Module):
    """
    G-buffer -> photorealistic style transfer.

    Pipeline:
      1. Conv encoder: 540×960 -> 270×480 -> 135×240 -> 67×120  (tokenize)
      2. Transformer blocks at 67×120 (global style understanding)
      3. Temporal GRU at 67×120 (temporal coherence, ~free)
      4. Conv decoder: 67×120 -> 135×240 -> 270×480 -> 540×960  (reconstruct)
      5. PixelShuffle: 540×960 -> 1080×1920  (upscale)

    ~800K params. Target: <10ms at 1080p on RTX 5060 Ti.
    """
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        enc = cfg.enc_channels   # [32, 64, 128]
        dec = cfg.dec_channels   # [128, 64, 32]

        # Input projection (full res)
        self.input_conv = nn.Sequential(
            nn.Conv2d(cfg.input_channels, enc[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder (downsample 3x = 8x total reduction)
        self.enc1 = DownBlock(enc[0], enc[1])   # /2: 270×480, 64ch
        self.enc2 = DownBlock(enc[1], enc[2])   # /4: 135×240, 128ch
        self.enc3 = DownBlock(enc[2], cfg.bottleneck_ch)  # /8: 67×120, 128ch

        # Transformer bottleneck (global context at 1/8 res)
        self.transformer = nn.Sequential(
            *[TransformerBlock(cfg.bottleneck_ch, cfg.n_heads, cfg.ffn_ratio)
              for _ in range(cfg.n_transformer_blocks)]
        )

        # Temporal at bottleneck resolution (very cheap: only 8K pixels)
        self.temporal = TemporalGRU(cfg.bottleneck_ch) if cfg.temporal == "gru" else None

        # Decoder (upsample with skip connections)
        self.dec3 = UpBlock(cfg.bottleneck_ch, enc[2], dec[0])  # ×2: 135×240
        self.dec2 = UpBlock(dec[0], enc[1], dec[1])              # ×2: 270×480
        self.dec1 = UpBlock(dec[1], enc[0], dec[2])              # ×2: 540×960

        # Output: dual PixelShuffle paths (2x and 3x)
        # Only ONE runs per frame — zero wasted compute
        self.to_rgb_2x = nn.Sequential(
            nn.Conv2d(dec[2], 3 * 4, 3, padding=1),   # 32 -> 12ch
            nn.PixelShuffle(2),
        )
        self.to_rgb_3x = nn.Sequential(
            nn.Conv2d(dec[2], 3 * 9, 3, padding=1),   # 32 -> 27ch
            nn.PixelShuffle(3),
        )

    def forward(self, color, depth, motion_vectors,
                prev_output=None, prev_hidden=None,
                target_h=0, target_w=0):
        B, _, rH, rW = color.shape
        if target_h <= 0: target_h = rH * self.cfg.scale
        if target_w <= 0: target_w = rW * self.cfg.scale

        # Build input
        inputs = [color, depth.to(color.dtype), motion_vectors]
        if self.cfg.use_warped_prev:
            if prev_output is not None:
                prev_down = F.interpolate(prev_output, (rH, rW), mode="bilinear", align_corners=False)
                inputs.append(warp(prev_down, motion_vectors))
            else:
                inputs.append(torch.zeros(B, 3, rH, rW, device=color.device, dtype=color.dtype))
        x = torch.cat(inputs, dim=1)

        # Encoder
        e0 = self.input_conv(x)          # 540×960, 32ch
        e1 = self.enc1(e0)               # 270×480, 64ch
        e2 = self.enc2(e1)               # 135×240, 128ch
        e3 = self.enc3(e2)               # 67×120, 128ch

        # Transformer at bottleneck (reshape to sequence)
        B, C, H, W = e3.shape
        tokens = e3.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        tokens = self.transformer(tokens)
        e3_out = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # Temporal at bottleneck resolution
        if self.temporal is not None:
            # Downsample motion vectors to bottleneck res
            mv_down = F.interpolate(motion_vectors, (H, W), mode="bilinear", align_corners=False)
            mv_down = mv_down * torch.tensor([W / rW, H / rH], device=mv_down.device).view(1, 2, 1, 1)
            hidden = self.temporal(e3_out, prev_hidden, mv_down)
        else:
            hidden = e3_out

        # Decoder with skip connections
        d2 = self.dec3(hidden, e2)        # 135×240, 128ch
        d1 = self.dec2(d2, e1)            # 270×480, 64ch
        d0 = self.dec1(d1, e0)            # 540×960, 32ch

        # Output — pick PixelShuffle path based on target scale
        scale = max(target_h / rH, target_w / rW)
        if scale <= 2.0:
            output = self.to_rgb_2x(d0)
        else:
            output = self.to_rgb_3x(d0)

        # Sigmoid in float32 to prevent channel death in bf16
        output = output.float().sigmoid().to(d0.dtype)

        if output.shape[2] != target_h or output.shape[3] != target_w:
            output = F.interpolate(output, (target_h, target_w), mode="bilinear", align_corners=False)

        return output, hidden.detach()


# ============================================================================
# Presets
# ============================================================================

PRESETS = {
    "fast": ModelConfig(
        enc_channels=[24, 48, 96], bottleneck_ch=96,
        n_transformer_blocks=2, n_heads=4, ffn_ratio=2,
        dec_channels=[96, 48, 24], temporal="ema",
    ),
    "balanced": ModelConfig(
        enc_channels=[32, 64, 128], bottleneck_ch=128,
        n_transformer_blocks=4, n_heads=4, ffn_ratio=4,
        dec_channels=[128, 64, 32], temporal="gru",
    ),
    "quality": ModelConfig(
        enc_channels=[48, 96, 192], bottleneck_ch=192,
        n_transformer_blocks=6, n_heads=6, ffn_ratio=4,
        dec_channels=[192, 96, 48], temporal="gru",
    ),
}


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        model = PrismV2(cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} ({params:,} params, {params/1e6:.2f}M) ===")
        print(f"    enc={cfg.enc_channels} bottleneck={cfg.bottleneck_ch} "
              f"transformer={cfg.n_transformer_blocks} heads={cfg.n_heads}")

        # Test forward pass at small resolution
        rH, rW = 30, 54
        out, hidden = model(
            torch.randn(1, 3, rH, rW),
            torch.randn(1, 1, rH, rW),
            torch.randn(1, 2, rH, rW),
        )
        print(f"    {rH}×{rW} -> {out.shape[2]}×{out.shape[3]} | hidden: {hidden.shape}")

        # Temporal test
        out2, _ = model(
            torch.randn(1, 3, rH, rW),
            torch.randn(1, 1, rH, rW),
            torch.randn(1, 2, rH, rW),
            prev_output=out.detach(),
            prev_hidden=hidden,
        )
        print(f"    Temporal frame 2: {out2.shape[2]}×{out2.shape[3]} OK")


# ============================================================================
# Discriminator — only used during training
# ============================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers = [sn(nn.Conv2d(in_ch, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, False)]
        ch = ndf
        for i in range(1, n_layers):
            ch_next = min(ndf * 2 ** i, ndf * 8)
            layers += [sn(nn.Conv2d(ch, ch_next, 4, 2, 1)), nn.LeakyReLU(0.2, False)]
            ch = ch_next
        layers.append(sn(nn.Conv2d(ch, 1, 4, 1, 1)))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.disc1 = PatchDiscriminator(in_ch)
        self.disc2 = PatchDiscriminator(in_ch)
        self.down = nn.AvgPool2d(3, 2, 1)
    def forward(self, x): return [self.disc1(x), self.disc2(self.down(x))]

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        for p in self.parameters(): p.requires_grad = False
    def forward(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss

class HingeLoss:
    @staticmethod
    def d_loss(real_preds, fake_preds):
        return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for r, f in zip(real_preds, fake_preds)) / len(real_preds)
    @staticmethod
    def g_loss(fake_preds):
        return sum(-f.mean() for f in fake_preds) / len(fake_preds)
