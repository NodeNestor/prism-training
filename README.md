# Prism Training -- Neural Style Transfer for Games

Training pipeline for [Prism](https://github.com/NodeNestor/prism-inference), a neural renderer that transforms game graphics toward photorealism in real-time.

**Status: Training in progress.** Models are actively being trained. First results are promising.

## What This Does

Prism takes a game's G-buffer (color, depth, motion vectors) and outputs a photorealistic version of the same scene. Not just upscaling -- full style transfer: lighting, materials, color grading, atmospheric effects.

Think DLSS 5 Ray Reconstruction, but open source and running on a single mid-range GPU.

## Architecture

**U-Net + Transformer (2.5M params, <12ms at 1080p)**

```
Game G-buffer (540p)
  -> Conv Encoder: 540p -> 270p -> 135p -> 68p
  -> 4x Windowed Transformer Blocks at 68p (global scene understanding)
  -> Temporal GRU at 68p (frame-to-frame coherence)
  -> Conv Decoder with skip connections: 68p -> 135p -> 270p -> 540p
  -> PixelShuffle: 540p -> 1080p photorealistic output
```

- **Encoder** downsamples 8x so the transformer processes only 8K tokens (not 518K pixels)
- **Windowed self-attention** (8x8) provides global context for style transfer
- **Temporal GRU** at bottleneck resolution maintains frame coherence (~free)
- **PixelShuffle** upscales 2x or 3x (supports all DLSS/FSR presets)
- **Inference**: 12.4ms on RTX 5060 Ti using Vulkan cooperative vectors (tensor cores)

## Training

GAN training with multi-scale PatchDiscriminator:

```bash
# Basic training
python train.py --data path/to/dataset --device cuda:0 --model balanced --amp --wandb

# Resume from checkpoint
python train.py --data path/to/dataset --resume checkpoints/checkpoint_ep50.pth --amp --wandb
```

### Training Strategy

- **Mixed real + synthetic data** (50/50 per batch)
- **Discriminator only sees real video as "real"** -- forces generator to learn photorealistic style
- **Game-style augmentation** on real video input (flat lighting, color banding, sharpening) -- teaches the model to reverse game artifacts
- **Streaming temporal** -- hidden state flows between frames (no BPTT, just forward)
- **Both 2x and 3x PixelShuffle** trained jointly

### Model Presets

| Preset | Params | Transformer | Target Speed |
|--------|--------|-------------|-------------|
| fast | 590K | 2 blocks | ~3ms @ 1080p |
| balanced | 2.5M | 4 blocks | ~12ms @ 1080p |
| quality | 6.4M | 6 blocks | ~20ms @ 1080p |

## Dataset Generation

Generate training data from video files:

```bash
# From video files (extracts frames + Depth Anything V2 + RAFT optical flow)
python generate_dataset.py --videos path/to/videos --output data/dataset --device cuda:0

# Batch generation from multiple sources
python generate_v3.py --device cuda:0 --output data/dataset
```

Each sample contains:
- **color** [3, 256, 256] -- render-resolution input (with game-style degradation for real video)
- **depth** [1, 256, 256] -- monocular depth (Depth Anything V2)
- **motion_vectors** [2, 256, 256] -- optical flow (RAFT)
- **ground_truth** [3, 512, 512] or [3, 768, 768] -- target output (2x or 3x)
- **is_real** -- True for real video, False for game/synthetic

### Data Sources

For best results, use diverse video covering:
- Nature (forests, water, mountains, caves)
- Urban (cities, streets, buildings)
- Medieval/fantasy (castles, villages -- directly game-relevant)
- Sci-fi/industrial (spaceships, factories)
- Action/FPS POV (first-person, fast motion)
- Faces (close-up, medium, far -- from both real and game sources)
- VFX/effects (particles, lighting, fire)

## wandb Logging

With `--wandb`, training logs:
- Loss curves (L1, adversarial, temporal, discriminator)
- Fixed sample comparisons every epoch: Nearest | Bilinear | Prism | GT (both 2x and 3x)
- Model architecture info

## Inference

See [prism-inference](https://github.com/NodeNestor/prism-inference) for the Vulkan compute inference engine with cooperative vectors (tensor cores).

## License

MIT
