"""
Generate V3 training dataset from ALL raw sources.
uint8 storage for 2x compression. Multiple crops per frame. Both 2x and 3x scales.

Runs Depth Anything V2 + RAFT on GPU for depth + optical flow.

Usage:
  python generate_v3.py --device cuda:0     # RTX 4060
  python generate_v3.py --device cuda:0 --quick  # 10% for testing
"""

import argparse, random, time, io, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# === Source definitions ===
# (target_samples, skip_frames, is_real)
SOURCES = {
    # REAL (discriminator sees these as "real")
    "pexels-nature":    (12000, 5,  True),
    "caves":            (6000,  5,  True),
    "medieval":         (10000, 3,  True),
    "scifi-industrial": (10000, 3,  True),
    "dl3dv-drone":      (5000,  10, True),
    "bdd100k":          (5000,  5,  True),
    "driving":          (3000,  5,  True),
    "fps-pov":          (8000,  3,  True),
    "sports-action":    (5000,  3,  True),
    "xd-violence":      (5000,  10, True),
    "effects":          (3000,  5,  True),
    "hdvila-filtered":  (3000,  5,  True),
    "finevideo":        (2000,  3,  True),
    "archive-war":      (2000,  5,  True),
    # SYNTHETIC/GAME (discriminator sees these as "fake")
    "game-screenshots-8k": (10000, 1, False),
}

RAW_DIR = Path("I:/prism-dataset/raw")
CROP_IN = 256
CROPS_PER_FRAME = 4  # random crops per extracted frame


def find_media(source_dir):
    """Find all video/image files."""
    vid_exts = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
    img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    videos, images = [], []
    for f in sorted(source_dir.rglob("*")):
        if f.suffix.lower() in vid_exts:
            videos.append(f)
        elif f.suffix.lower() in img_exts:
            images.append(f)
    return videos, images


def extract_video_frames(video_path, max_frames, skip):
    """Extract frames from video, skipping every N."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % (skip + 1) == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def save_sample_uint8(sample, path):
    """Save with uint8 color/GT for 2x compression."""
    compressed = {}
    for k, v in sample.items():
        if k in ('color', 'ground_truth') and v.dtype == torch.float16:
            compressed[k] = (v.float().clamp(0, 1) * 255).byte()
        else:
            compressed[k] = v
    torch.save(compressed, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="E:/prism-v3-train")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--source", type=str, default=None, help="Process only this source")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(exist_ok=True)

    # Count existing
    existing = len(list(output.glob("*.pt")))
    out_idx = existing
    print(f"Output: {output} ({existing} existing)")

    # Load depth + flow models
    print(f"Loading models on {args.device}...")
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dataset import DepthEstimator, FlowEstimator, get_jitter
    from game_stylize import apply_game_style

    depth_est = DepthEstimator(args.device)
    flow_est = FlowEstimator(args.device)
    rng_np = np.random.default_rng(42)
    rng = random.Random(42)

    sources = SOURCES
    if args.source:
        sources = {args.source: SOURCES[args.source]}

    total_generated = 0
    t_start = time.time()

    for name, (target, skip, is_real) in sources.items():
        src_dir = RAW_DIR / name
        if not src_dir.exists():
            print(f"\nSkip {name} (not found)")
            continue

        if args.quick:
            target = max(target // 10, 100)

        videos, images = find_media(src_dir)
        media_count = len(videos) + len(images)
        if media_count == 0:
            print(f"\nSkip {name} (no media)")
            continue

        print(f"\n{'='*50}")
        print(f"{name}: {len(videos)} videos, {len(images)} images, target={target}, is_real={is_real}")

        source_count = 0
        frames_needed = target // CROPS_PER_FRAME + 1

        # Process videos
        if videos:
            random.shuffle(videos)
            frames_per_vid = max(2, frames_needed // len(videos))

            for vid in videos:
                if source_count >= target:
                    break
                try:
                    frames = extract_video_frames(vid, frames_per_vid, skip)
                    if len(frames) < 2:
                        continue

                    prev_frame = None
                    for fi, frame in enumerate(frames):
                        if source_count >= target:
                            break

                        h, w = frame.shape[:2]
                        if h < 480 or w < 640:
                            prev_frame = frame
                            continue

                        # Compute depth + flow at display res
                        try:
                            depth_map = depth_est.predict(frame)
                            if prev_frame is not None:
                                flow = flow_est.predict(prev_frame, frame)
                            else:
                                flow = np.zeros((h, w, 2), dtype=np.float32)
                        except:
                            prev_frame = frame
                            continue

                        # Multiple random crops
                        for ci in range(CROPS_PER_FRAME):
                            if source_count >= target:
                                break

                            # Random scale: 2x or 3x
                            scale = rng.choice([2, 3])
                            crop_gt = CROP_IN * scale

                            if h < crop_gt or w < crop_gt:
                                continue

                            # Random crop position (in GT/display space)
                            cy = rng.randint(0, h - crop_gt)
                            cx = rng.randint(0, w - crop_gt)

                            # GT crop
                            gt_crop = frame[cy:cy+crop_gt, cx:cx+crop_gt].astype(np.float32) / 255.0

                            # Input = downsample GT crop
                            color_crop = cv2.resize(gt_crop, (CROP_IN, CROP_IN), interpolation=cv2.INTER_AREA)

                            # Apply game-style degradation to real video input
                            if is_real:
                                color_crop = apply_game_style(color_crop, style="random", rng=rng_np)

                            # Depth crop + downsample
                            depth_crop = depth_map[cy:cy+crop_gt, cx:cx+crop_gt]
                            depth_crop = cv2.resize(depth_crop, (CROP_IN, CROP_IN), interpolation=cv2.INTER_LINEAR)

                            # Flow crop + downsample + scale
                            flow_crop = flow[cy:cy+crop_gt, cx:cx+crop_gt].copy()
                            flow_crop = cv2.resize(flow_crop, (CROP_IN, CROP_IN), interpolation=cv2.INTER_LINEAR)
                            flow_crop /= scale  # scale motion vectors to input res

                            # Jitter
                            jx, jy = get_jitter(fi * CROPS_PER_FRAME + ci)

                            sample = {
                                'color': torch.from_numpy(color_crop).permute(2,0,1).half(),
                                'depth': torch.from_numpy(depth_crop).unsqueeze(0).half(),
                                'motion_vectors': torch.from_numpy(flow_crop).permute(2,0,1).half(),
                                'ground_truth': torch.from_numpy(gt_crop).permute(2,0,1).half(),
                                'is_real': torch.tensor(is_real, dtype=torch.bool),
                                'scale': torch.tensor(scale, dtype=torch.int32),
                            }

                            save_sample_uint8(sample, output / f"sample_{out_idx:06d}.pt")
                            out_idx += 1
                            source_count += 1

                        prev_frame = frame

                except Exception as e:
                    continue

        # Process images (game screenshots etc)
        if images and source_count < target:
            random.shuffle(images)
            for img_path in images:
                if source_count >= target:
                    break
                try:
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    if h < 512 or w < 512:
                        continue

                    depth_map = depth_est.predict(frame)
                    flow = np.zeros((h, w, 2), dtype=np.float32)

                    for ci in range(CROPS_PER_FRAME):
                        if source_count >= target:
                            break
                        scale = rng.choice([2, 3])
                        crop_gt = CROP_IN * scale
                        if h < crop_gt or w < crop_gt:
                            continue

                        cy = rng.randint(0, h - crop_gt)
                        cx = rng.randint(0, w - crop_gt)

                        gt_crop = frame[cy:cy+crop_gt, cx:cx+crop_gt].astype(np.float32) / 255.0
                        color_crop = cv2.resize(gt_crop, (CROP_IN, CROP_IN), interpolation=cv2.INTER_AREA)
                        depth_crop = cv2.resize(depth_map[cy:cy+crop_gt, cx:cx+crop_gt], (CROP_IN, CROP_IN))
                        flow_crop = np.zeros((CROP_IN, CROP_IN, 2), dtype=np.float32)

                        sample = {
                            'color': torch.from_numpy(color_crop).permute(2,0,1).half(),
                            'depth': torch.from_numpy(depth_crop).unsqueeze(0).half(),
                            'motion_vectors': torch.from_numpy(flow_crop).permute(2,0,1).half(),
                            'ground_truth': torch.from_numpy(gt_crop).permute(2,0,1).half(),
                            'is_real': torch.tensor(is_real, dtype=torch.bool),
                            'scale': torch.tensor(scale, dtype=torch.int32),
                        }
                        save_sample_uint8(sample, output / f"sample_{out_idx:06d}.pt")
                        out_idx += 1
                        source_count += 1

                except:
                    continue

        total_generated += source_count
        elapsed = time.time() - t_start
        rate = total_generated / elapsed if elapsed > 0 else 0
        print(f"  {source_count} samples from {name} | total: {total_generated} | {rate:.1f} samples/sec")

    # Final stats
    elapsed = time.time() - t_start
    total_size = sum(f.stat().st_size for f in list(output.glob("*.pt"))[:1000])
    avg_size = total_size / min(1000, out_idx) if out_idx > 0 else 0
    total_gb = avg_size * out_idx / 1e9

    print(f"\n{'='*50}")
    print(f"DONE: {out_idx} samples in {elapsed/60:.0f} min ({elapsed/3600:.1f} hours)")
    print(f"Avg sample: {avg_size/1024:.0f} KB, total: {total_gb:.0f} GB")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
