"""
GPU-side game-style augmentation — applied on-the-fly during training.
No disk I/O, no CPU bottleneck. Just tensor ops on GPU.

Applied to the COLOR input only (not ground truth).
Random strength per batch = infinite variety.
"""

import torch
import torch.nn.functional as F


def quantize_gpu(x: torch.Tensor, levels: int = 32) -> torch.Tensor:
    """Reduce color precision — simulates low texture bit depth."""
    return (x * levels).round() / levels


def flatten_lighting_gpu(x: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Reduce lighting contrast — games have flatter ambient."""
    gray = x.mean(dim=1, keepdim=True)
    mean_lum = gray.mean(dim=(2, 3), keepdim=True)
    return x + strength * (mean_lum - gray)


def sharpen_gpu(x: torch.Tensor, amount: float = 0.5) -> torch.Tensor:
    """Over-sharpen — game renders have unnaturally sharp edges."""
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
    return x + amount * (x - blur)


def add_banding_gpu(x: torch.Tensor, bands: int = 48) -> torch.Tensor:
    """Color banding in gradients."""
    return (x * bands).floor() / bands


def gamma_shift_gpu(x: torch.Tensor, gamma: float = 1.1) -> torch.Tensor:
    """Slight gamma shift — different tone mapping."""
    return x.clamp(1e-6, 1.0).pow(gamma)


def smooth_skin_gpu(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Bilateral-like smoothing — makes textures look CG/plastic."""
    k = int(sigma * 4) | 1
    if k < 3:
        k = 3
    pad = k // 2
    channels = x.shape[1]
    kernel_1d = torch.exp(-torch.arange(k, device=x.device, dtype=x.dtype).sub(pad).pow(2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_h = kernel_1d.view(1, 1, k, 1).expand(channels, 1, k, 1)
    kernel_w = kernel_1d.view(1, 1, 1, k).expand(channels, 1, 1, k)
    blurred = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'), kernel_h, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='reflect'), kernel_w, groups=channels)
    return blurred


def oversaturate_gpu(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Crank saturation up — games love vivid unrealistic colors."""
    gray = x.mean(dim=1, keepdim=True)
    return (x - gray) * (1 + strength) + gray


def color_tint_gpu(x: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Random color tint — games have weird white balance / color grading."""
    B, C, H, W = x.shape
    # Random per-channel multiplier
    tint = torch.empty(B, 3, 1, 1, device=x.device, dtype=x.dtype).uniform_(1 - strength, 1 + strength)
    return x * tint


def channel_swap_gpu(x: torch.Tensor) -> torch.Tensor:
    """Randomly shift color channels — simulates wrong color space / weird grading."""
    B, C, H, W = x.shape
    shift = torch.empty(B, 3, 1, 1, device=x.device, dtype=x.dtype).uniform_(-0.08, 0.08)
    return x + shift


def crush_blacks_gpu(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Lift or crush blacks — games often have no true black or clip shadows."""
    lift = strength * 0.15  # lift black point
    return x * (1 - lift) + lift


def invert_partial_gpu(x: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Partial inversion on random channel — creates unnatural color look."""
    ch = torch.randint(0, 3, (1,)).item()
    out = x.clone()
    out[:, ch] = out[:, ch] * (1 - strength) + (1 - out[:, ch]) * strength
    return out


def hue_rotate_gpu(x: torch.Tensor, angle: float = 0.1) -> torch.Tensor:
    """Approximate hue rotation — shifts colors around the wheel."""
    # Simple RGB rotation matrix approximation
    cos_a = torch.cos(torch.tensor(angle * 3.14159 * 2))
    sin_a = torch.sin(torch.tensor(angle * 3.14159 * 2))
    # Simplified hue rotation in RGB space
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    avg = (r + g + b) / 3
    r_out = avg + (r - avg) * cos_a + (b - g) * sin_a * 0.577
    g_out = avg + (g - avg) * cos_a + (r - b) * sin_a * 0.577
    b_out = avg + (b - avg) * cos_a + (g - r) * sin_a * 0.577
    return torch.cat([r_out, g_out, b_out], dim=1)


def tone_map_gpu(x: torch.Tensor, style: int = 0) -> torch.Tensor:
    """Different tone mapping curves — filmic, ACES, linear, etc."""
    if style == 0:
        # Filmic S-curve — crush shadows, roll highlights
        return (x * x * (3 - 2 * x))
    elif style == 1:
        # ACES-like — contrasty mid, soft highlights
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0, 1)
    else:
        # Harsh linear with knee
        return torch.where(x < 0.5, x * 1.3, 0.65 + (x - 0.5) * 0.7)


def noise_gpu(x: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    """Add noise — simulates sensor noise or dithering artifacts."""
    return x + torch.randn_like(x) * strength


def posterize_gpu(x: torch.Tensor, levels: int = 6) -> torch.Tensor:
    """Posterize — cel-shade look, flattens into solid color regions."""
    return (x * levels).floor() / levels


def kill_detail_gpu(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Remove fine detail — makes textures look painted/CG.
    Blurs then re-sharpens edges, killing small texture detail but keeping structure."""
    # Strong blur to kill detail
    k = 7
    pad = k // 2
    channels = x.shape[1]
    kernel_1d = torch.ones(k, device=x.device, dtype=x.dtype) / k
    kernel_h = kernel_1d.view(1, 1, k, 1).expand(channels, 1, k, 1)
    kernel_w = kernel_1d.view(1, 1, 1, k).expand(channels, 1, 1, k)
    blurred = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'), kernel_h, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='reflect'), kernel_w, groups=channels)
    return x * (1 - strength) + blurred * strength


def desaturate_gpu(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Reduce saturation — games often have less natural color richness."""
    gray = x.mean(dim=1, keepdim=True)
    return x * (1 - strength) + gray * strength


def hard_shadows_gpu(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Push darks darker and lights lighter — hard shadow look, no soft falloff."""
    # Simple contrast boost that mimics hard shadow boundaries
    mid = 0.5
    return ((x - mid) * (1 + strength) + mid).clamp(0, 1)


def reduce_color_range_gpu(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Compress dynamic range — games often have limited color range."""
    lo = 0.1 * strength
    hi = 1.0 - 0.1 * strength
    return x * (hi - lo) + lo


def game_augment_gpu(
    color: torch.Tensor,
    strength: float = 0.5,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply random game-style degradation to color tensor on GPU.

    Args:
        color: [B, 3, H, W] float tensor in [0, 1]
        strength: 0.0 = no augment, 1.0 = maximum game-ification
        rng: torch random generator for reproducibility

    Returns:
        Augmented color tensor, same shape
    """
    if strength <= 0:
        return color

    def rand():
        return torch.rand(1, generator=rng).item()

    x = color

    # Always apply at least 2-3 effects. Higher probability + stronger params.

    # Flatten lighting — games have flat ambient
    if rand() < 0.7:
        x = flatten_lighting_gpu(x, strength=0.2 + rand() * 0.6 * strength)

    # Posterize / cel-shade — kills detail into flat regions
    if rand() < 0.5:
        levels = int(8 + (1 - strength) * 16)  # 8-24 levels (not too extreme)
        x = posterize_gpu(x, levels=levels)

    # Kill fine texture detail
    if rand() < 0.5:
        x = kill_detail_gpu(x, strength=0.3 + rand() * 0.5 * strength)

    # Over-sharpen edges
    if rand() < 0.6:
        x = sharpen_gpu(x, amount=0.3 + rand() * 1.0 * strength)

    # Color quantize — low bit depth textures
    if rand() < 0.5:
        levels = int(8 + (1 - strength) * 24)
        x = quantize_gpu(x, levels=levels)

    # Color banding
    if rand() < 0.4:
        bands = int(12 + (1 - strength) * 32)
        x = add_banding_gpu(x, bands=bands)

    # Gamma shift — slight tone mapping difference (only brighten, never crush darks)
    if rand() < 0.3:
        x = gamma_shift_gpu(x, gamma=0.8 + rand() * 0.2)  # 0.8-1.0 = slight brighten

    # Desaturate — less natural color
    if rand() < 0.4:
        x = desaturate_gpu(x, strength=0.1 + rand() * 0.3 * strength)

    # Hard shadows — no soft falloff (keep mild, don't crush)
    if rand() < 0.3:
        x = hard_shadows_gpu(x, strength=0.1 + rand() * 0.3 * strength)

    # Reduce dynamic range (keep mild)
    if rand() < 0.2:
        x = reduce_color_range_gpu(x, strength=0.1 + rand() * 0.3 * strength)

    # CG plastic smooth
    if rand() < 0.4:
        x = smooth_skin_gpu(x, sigma=1.0 + rand() * 2.0 * strength)

    # --- Color fuckery ---

    # Oversaturate — games love cranked colors
    if rand() < 0.5:
        x = oversaturate_gpu(x, strength=0.3 + rand() * 0.7 * strength)

    # Random color tint — wrong white balance
    if rand() < 0.5:
        x = color_tint_gpu(x, strength=0.1 + rand() * 0.3 * strength)

    # Channel offset — shifted color space
    if rand() < 0.3:
        x = channel_swap_gpu(x)

    # Crush/lift blacks
    if rand() < 0.4:
        x = crush_blacks_gpu(x, strength=0.2 + rand() * 0.5 * strength)

    # Partial channel inversion — weird unnatural color
    if rand() < 0.2:
        x = invert_partial_gpu(x, strength=0.1 + rand() * 0.2 * strength)

    # Hue rotation — shift colors around
    if rand() < 0.3:
        x = hue_rotate_gpu(x, angle=rand() * 0.15 * strength)

    # Different tone mapping curve
    if rand() < 0.3:
        style = int(rand() * 3)
        x = tone_map_gpu(x, style=style)

    # Noise / dithering
    if rand() < 0.3:
        x = noise_gpu(x, strength=0.02 + rand() * 0.06 * strength)

    return x.clamp(0, 1)


if __name__ == "__main__":
    # Quick test
    x = torch.randn(1, 3, 128, 128).clamp(0, 1)
    for s in [0.0, 0.3, 0.5, 0.8, 1.0]:
        out = game_augment_gpu(x, strength=s)
        diff = (x - out).abs().mean()
        print(f"strength={s}: diff={diff:.4f} min={out.min():.3f} max={out.max():.3f}")
    print("GPU augmentation OK!")
