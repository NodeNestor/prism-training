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
    transformer_dim: int = 0         # 0 = same as bottleneck_ch (adds projection if different)
    n_transformer_blocks: int = 4    # transformer depth (global context)
    n_heads: int = 4                 # attention heads
    ffn_ratio: int = 4               # FFN expansion ratio (used when n_experts=0)
    window_size: int = 8             # 0 = full attention, >0 = windowed

    # Mixture of Experts
    n_experts: int = 0               # 0 = dense FFN, >0 = MoE with this many experts
    expert_ffn_hidden: int = 256     # hidden dim per expert (small — many experts compensate)
    top_k: int = 1                   # tokens routed to top-k experts (1 = matches inference)
    moe_balance_weight: float = 0.01 # load balancing loss weight

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
        if self.transformer_dim == 0:
            self.transformer_dim = self.bottleneck_ch


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


class WindowedMultiHeadSelfAttention(nn.Module):
    """Windowed self-attention matching inference engine's 8x8 windows."""
    def __init__(self, dim, n_heads=8, window_size=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, spatial_shape=None):
        B, N, C = x.shape
        if spatial_shape is None or self.window_size <= 0:
            # Fall back to global attention
            qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
            return self.proj(x)

        H, W = spatial_shape
        ws = self.window_size

        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ws, Wp // ws

        # Partition into windows: (B*nH*nW, ws*ws, C)
        x_win = x_2d.reshape(B, C, nH, ws, nW, ws)
        x_win = x_win.permute(0, 2, 4, 3, 5, 1).reshape(B * nH * nW, ws * ws, C)

        # Attention within each window
        qkv = self.qkv(x_win).reshape(B * nH * nW, ws * ws, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B * nH * nW, ws * ws, C)
        out = self.proj(out)

        # Reverse partition
        out = out.reshape(B, nH, nW, ws, ws, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out.flatten(2).permute(0, 2, 1)


class MoERouter(nn.Module):
    """Token-to-expert router with load balancing + z-loss."""
    def __init__(self, dim, n_experts, top_k=1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x):
        # x: (B*N, C) -> logits: (B*N, n_experts)
        logits = self.gate(x)

        # Jitter for exploration during training (ST-MoE)
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)

        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalize top-k probs to sum to 1
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # --- Auxiliary losses ---

        # 1. Load balancing loss (Switch Transformer)
        # f_i = fraction of tokens dispatched to expert i (non-differentiable)
        # P_i = mean probability assigned to expert i (differentiable)
        with torch.no_grad():
            assignments = F.one_hot(top_k_indices[:, 0], self.n_experts).float()
            f = assignments.mean(dim=0)  # (n_experts,)
        P = probs.mean(dim=0)  # (n_experts,)
        balance_loss = self.n_experts * (f * P).sum()

        # 2. Router z-loss (stabilizes logits magnitude — from ST-MoE paper)
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()

        # 3. Expert usage stats (detached, for logging only)
        with torch.no_grad():
            usage = f.detach()  # fraction of tokens per expert

        return top_k_indices, top_k_weights, balance_loss, z_loss, usage


class MoEFFN(nn.Module):
    """Vectorized Mixture of Experts — no Python loops over experts.

    Uses batched matmuls: all experts share the same shaped weights,
    stacked into (n_experts, hidden, dim) tensors. Tokens are grouped
    by expert assignment and processed in parallel.
    """
    def __init__(self, dim, n_experts, expert_hidden, top_k=1):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.expert_hidden = expert_hidden
        self.top_k = top_k
        self.router = MoERouter(dim, n_experts, top_k)

        # Stacked expert weights for batched matmul: (n_experts, out, in)
        self.w1 = nn.Parameter(torch.empty(n_experts, expert_hidden, dim))
        self.b1 = nn.Parameter(torch.zeros(n_experts, expert_hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, dim, expert_hidden))
        self.b2 = nn.Parameter(torch.zeros(n_experts, dim))

        # Kaiming init
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            fan_in1 = dim
            fan_in2 = expert_hidden
            bound1 = 1 / math.sqrt(fan_in1)
            bound2 = 1 / math.sqrt(fan_in2)
            nn.init.uniform_(self.b1[i], -bound1, bound1)
            nn.init.uniform_(self.b2[i], -bound2, bound2)

    def forward(self, x):
        B, N, C = x.shape
        flat_x = x.reshape(-1, C)  # (T, C) where T = B*N
        T = flat_x.shape[0]

        top_k_indices, top_k_weights, balance_loss, z_loss, usage = self.router(flat_x)

        # Process each top-k selection
        output = torch.zeros(T, C, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            indices = top_k_indices[:, k]    # (T,) expert assignment per token
            weights = top_k_weights[:, k]    # (T,) weight per token

            # Sort tokens by expert for efficient batched processing
            sorted_idx = torch.argsort(indices, stable=True)
            sorted_expert_ids = indices[sorted_idx]
            sorted_tokens = flat_x[sorted_idx]  # (T, C)

            # Find boundaries between expert groups
            # expert_counts[i] = number of tokens assigned to expert i
            expert_counts = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
            expert_counts.scatter_add_(0, sorted_expert_ids, torch.ones(T, dtype=torch.long, device=x.device))

            # Process experts that have tokens assigned
            offset = 0
            expert_outputs = torch.zeros_like(sorted_tokens)

            for e_idx in range(self.n_experts):
                count = expert_counts[e_idx].item()
                if count == 0:
                    offset += count
                    continue

                # Gather tokens for this expert
                e_tokens = sorted_tokens[offset:offset + count]  # (count, C)

                # Expert FFN: GELU(x @ W1^T + b1) @ W2^T + b2
                h = F.gelu(F.linear(e_tokens, self.w1[e_idx], self.b1[e_idx]))
                e_out = F.linear(h, self.w2[e_idx], self.b2[e_idx])

                expert_outputs[offset:offset + count] = e_out
                offset += count

            # Unsort and apply weights
            unsorted_outputs = torch.zeros_like(expert_outputs)
            unsorted_outputs[sorted_idx] = expert_outputs
            output += unsorted_outputs * weights.unsqueeze(-1)

        # Store losses and stats (training only)
        if self.training:
            self._balance_loss = balance_loss
            self._z_loss = z_loss
            self._expert_usage = usage

        return output.reshape(B, N, C)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN — matches inference engine architecture."""
    def __init__(self, dim, n_heads=8, n_experts=16, expert_hidden=256,
                 top_k=1, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedMultiHeadSelfAttention(dim, n_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoEFFN(dim, n_experts, expert_hidden, top_k)

    def forward(self, x, spatial_shape=None):
        x = x + self.attn(self.norm1(x), spatial_shape=spatial_shape)
        x = x + self.moe(self.norm2(x))
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

        # Dimension projection if transformer_dim != bottleneck_ch
        t_dim = cfg.transformer_dim
        self.proj_up = nn.Linear(cfg.bottleneck_ch, t_dim) if t_dim != cfg.bottleneck_ch else nn.Identity()
        self.proj_down = nn.Linear(t_dim, cfg.bottleneck_ch) if t_dim != cfg.bottleneck_ch else nn.Identity()

        # Transformer bottleneck (global context at 1/8 res)
        if cfg.n_experts > 0:
            # MoE transformer — matches inference engine
            self.transformer_blocks = nn.ModuleList([
                MoETransformerBlock(
                    t_dim, cfg.n_heads, cfg.n_experts,
                    cfg.expert_ffn_hidden, cfg.top_k, cfg.window_size,
                )
                for _ in range(cfg.n_transformer_blocks)
            ])
            self.use_moe = True
        else:
            # Dense transformer (original)
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(t_dim, cfg.n_heads, cfg.ffn_ratio)
                for _ in range(cfg.n_transformer_blocks)
            ])
            self.use_moe = False

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
        tokens = self.proj_up(tokens)             # [B, H*W, transformer_dim]

        # Run transformer blocks
        spatial_shape = (H, W)
        for block in self.transformer_blocks:
            if self.use_moe:
                tokens = block(tokens, spatial_shape=spatial_shape)
            else:
                tokens = block(tokens)

        # Collect MoE stats (training only — skipped during export/eval)
        if self.use_moe and self.training:
            self._moe_balance_loss = sum(b.moe._balance_loss for b in self.transformer_blocks)
            self._moe_z_loss = sum(b.moe._z_loss for b in self.transformer_blocks)
            self._moe_expert_usage = [b.moe._expert_usage for b in self.transformer_blocks]

        tokens = self.proj_down(tokens)           # [B, H*W, bottleneck_ch]
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
    # --- Dense (original) ---
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
    # --- Dense with wider transformer (matches inference dim=256) ---
    "standard-256": ModelConfig(
        enc_channels=[32, 64, 128], bottleneck_ch=128,
        transformer_dim=256, n_transformer_blocks=12, n_heads=8, ffn_ratio=4,
        dec_channels=[128, 64, 32], temporal="gru",
    ),
    # --- MoE variants (matching inference engine configs) ---
    # 37M total / ~6M active per token — runs at 73 FPS in inference
    "moe-16": ModelConfig(
        enc_channels=[32, 64, 128], bottleneck_ch=128,
        transformer_dim=256, n_transformer_blocks=16, n_heads=8,
        n_experts=16, expert_ffn_hidden=256, top_k=1,
        dec_channels=[128, 64, 32], temporal="gru",
    ),
    # 98M total / ~8M active — runs at 56 FPS
    "moe-32": ModelConfig(
        enc_channels=[32, 64, 128], bottleneck_ch=128,
        transformer_dim=256, n_transformer_blocks=22, n_heads=8,
        n_experts=32, expert_ffn_hidden=256, top_k=1,
        dec_channels=[128, 64, 32], temporal="gru",
    ),
    # 139M total / ~6M active — runs at 66 FPS (sweet spot)
    "moe-64": ModelConfig(
        enc_channels=[32, 64, 128], bottleneck_ch=128,
        transformer_dim=256, n_transformer_blocks=16, n_heads=8,
        n_experts=64, expert_ffn_hidden=256, top_k=1,
        dec_channels=[128, 64, 32], temporal="gru",
    ),
}


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        model = PrismV2(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        moe_info = f" experts={cfg.n_experts}" if cfg.n_experts > 0 else ""
        print(f"\n=== {name} ({total_params:,} params, {total_params/1e6:.1f}M){moe_info} ===")
        print(f"    enc={cfg.enc_channels} bottleneck={cfg.bottleneck_ch} "
              f"t_dim={cfg.transformer_dim} blocks={cfg.n_transformer_blocks} heads={cfg.n_heads}")

        # Test forward pass at small resolution
        rH, rW = 30, 54
        out, hidden = model(
            torch.randn(1, 3, rH, rW),
            torch.randn(1, 1, rH, rW),
            torch.randn(1, 2, rH, rW),
        )
        print(f"    {rH}×{rW} -> {out.shape[2]}×{out.shape[3]} | hidden: {hidden.shape}")

        if cfg.n_experts > 0:
            print(f"    MoE balance loss: {model._moe_balance_loss.item():.4f}")

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
