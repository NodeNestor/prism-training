"""
Generate SYNTHETIC/GAME samples to balance the dataset 50/50 with real.

Sources:
  - TartanAir (539GB raw, diverse 3D environments with ground-truth depth+flow)
  - game-screenshots-8k (55GB, real game screenshots)
  - FFHQ faces (7GB, face images for face diversity)

These get is_real=False so the discriminator learns to distinguish game vs real.
Except FFHQ which gets is_real=True (real faces, just from images not video).

Output goes to I:/prism-dataset/dataset-v3/ starting from the next available index.

Usage:
  python generate_synthetic.py --device cuda:0
"""

import argparse, random, time, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

CROP_IN = 256
CROPS_PER_FRAME = 4
RAW_DIR = Path("I:/prism-dataset/raw")


def save_sample_uint8(sample, path):
    compressed = {}
    for k, v in sample.items():
        if k in ('color', 'ground_truth') and v.dtype == torch.float16:
            compressed[k] = (v.float().clamp(0, 1) * 255).byte()
        else:
            compressed[k] = v
    torch.save(compressed, path)


def find_media(source_dir, exts=None):
    if exts is None:
        exts = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
    files = []
    for f in sorted(source_dir.rglob("*")):
        if f.suffix.lower() in exts:
            files.append(f)
    return files


def extract_video_frames(video_path, max_frames, skip):
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


def process_tartanair(output_dir, start_idx, target, device):
    """Process TartanAir — has pre-computed depth, just need to extract frames."""
    tartanair_dir = RAW_DIR / "tartanair"
    if not tartanair_dir.exists():
        print(f"TartanAir not found at {tartanair_dir}")
        return start_idx

    # TartanAir is stored as tarballs or extracted folders
    # Check for extracted image sequences
    envs = []
    for d in sorted(tartanair_dir.rglob("*")):
        if d.is_dir() and any(d.glob("*.png")):
            envs.append(d)

    if not envs:
        # Try to find video files instead
        videos = find_media(tartanair_dir)
        if videos:
            print(f"TartanAir: {len(videos)} videos found")
        else:
            print(f"TartanAir: no extracted data or videos found, skipping")
            return start_idx

    print(f"TartanAir: {len(envs)} environments found")
    # Process will be handled by main loop
    return start_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="I:/prism-dataset/dataset-v3")
    parser.add_argument("--target-synthetic", type=int, default=45000,
                        help="Target synthetic samples (to match ~88K real)")
    parser.add_argument("--target-faces", type=int, default=10000)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Find starting index
    existing = sorted(output.glob("sample_*.pt"))
    if existing:
        last_idx = int(existing[-1].stem.split("_")[1])
        out_idx = last_idx + 1
    else:
        out_idx = 0
    print(f"Output: {output} (starting at index {out_idx})")

    # Load depth + flow models
    print(f"Loading models on {args.device}...")
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dataset import DepthEstimator, FlowEstimator, get_jitter

    depth_est = DepthEstimator(args.device)
    flow_est = FlowEstimator(args.device)
    rng = random.Random(42)

    total_generated = 0
    t_start = time.time()

    # === 1. GAME SCREENSHOTS (is_real=False) ===
    game_dir = RAW_DIR / "game-screenshots-8k"
    if game_dir.exists():
        images = find_media(game_dir, {".png", ".jpg", ".jpeg", ".bmp"})
        target_game = min(args.target_synthetic // 3, len(images) * CROPS_PER_FRAME)
        print(f"\n{'='*50}")
        print(f"Game screenshots: {len(images)} images, target={target_game}")

        count = 0
        rng_imgs = list(images)
        random.shuffle(rng_imgs)
        prev_frame = None

        for img_path in tqdm(rng_imgs, desc="Game screenshots"):
            if count >= target_game:
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
                    if count >= target_game:
                        break
                    scale = rng.choice([2, 2, 3])  # favor 2x slightly
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
                        'color': torch.from_numpy(color_crop).permute(2, 0, 1).half(),
                        'depth': torch.from_numpy(depth_crop).unsqueeze(0).half(),
                        'motion_vectors': torch.from_numpy(flow_crop).permute(2, 0, 1).half(),
                        'ground_truth': torch.from_numpy(gt_crop).permute(2, 0, 1).half(),
                        'is_real': torch.tensor(False, dtype=torch.bool),
                        'scale': torch.tensor(scale, dtype=torch.int32),
                    }
                    save_sample_uint8(sample, output / f"sample_{out_idx:06d}.pt")
                    out_idx += 1
                    count += 1
            except Exception as e:
                continue

        total_generated += count
        print(f"  Generated {count} game screenshot samples")

    # === 2. FFHQ FACES (is_real=True — these ARE real faces) ===
    ffhq_dir = RAW_DIR / "ffhq-256"
    if ffhq_dir.exists():
        images = find_media(ffhq_dir / "data" if (ffhq_dir / "data").exists() else ffhq_dir,
                           {".png", ".jpg", ".jpeg"})
        target_faces = min(args.target_faces, len(images) * 2)
        print(f"\n{'='*50}")
        print(f"FFHQ faces: {len(images)} images, target={target_faces}")

        count = 0
        random.shuffle(images)
        for img_path in tqdm(images, desc="FFHQ faces"):
            if count >= target_faces:
                break
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                # FFHQ is 256x256 — upscale to get a larger canvas for cropping
                if h < 512:
                    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                    h, w = 512, 512

                depth_map = depth_est.predict(frame)

                for ci in range(2):  # 2 crops per face
                    if count >= target_faces:
                        break
                    scale = 2
                    crop_gt = CROP_IN * scale
                    if h < crop_gt or w < crop_gt:
                        continue

                    cy = rng.randint(0, h - crop_gt)
                    cx = rng.randint(0, w - crop_gt)

                    gt_crop = frame[cy:cy+crop_gt, cx:cx+crop_gt].astype(np.float32) / 255.0
                    color_crop = cv2.resize(gt_crop, (CROP_IN, CROP_IN), interpolation=cv2.INTER_AREA)
                    depth_crop = cv2.resize(depth_map[cy:cy+crop_gt, cx:cx+crop_gt], (CROP_IN, CROP_IN))

                    sample = {
                        'color': torch.from_numpy(color_crop).permute(2, 0, 1).half(),
                        'depth': torch.from_numpy(depth_crop).unsqueeze(0).half(),
                        'motion_vectors': torch.zeros(2, CROP_IN, CROP_IN, dtype=torch.float16),
                        'ground_truth': torch.from_numpy(gt_crop).permute(2, 0, 1).half(),
                        'is_real': torch.tensor(True, dtype=torch.bool),  # real faces!
                        'scale': torch.tensor(scale, dtype=torch.int32),
                    }
                    save_sample_uint8(sample, output / f"sample_{out_idx:06d}.pt")
                    out_idx += 1
                    count += 1
            except:
                continue

        total_generated += count
        print(f"  Generated {count} face samples")

    # === 3. MORE REAL VIDEO with better 3x balance ===
    # Process sources that are underrepresented
    underrepresented = {
        "scifi-industrial": (15000, 3, True),   # cool tones
        "xd-violence":      (10000, 10, True),  # urban/action
        "effects":          (5000, 5, True),     # VFX
        "kinetics-700":     (10000, 10, True),   # people/faces in video
    }

    for name, (target, skip, is_real) in underrepresented.items():
        src_dir = RAW_DIR / name
        if not src_dir.exists():
            print(f"\nSkip {name} (not found)")
            continue

        videos = find_media(src_dir)
        if not videos:
            print(f"\nSkip {name} (no videos)")
            continue

        print(f"\n{'='*50}")
        print(f"{name}: {len(videos)} videos, target={target}")

        count = 0
        random.shuffle(videos)
        frames_per_vid = max(2, (target // CROPS_PER_FRAME) // len(videos))

        for vid in tqdm(videos, desc=name):
            if count >= target:
                break
            try:
                frames = extract_video_frames(vid, frames_per_vid, skip)
                if len(frames) < 2:
                    continue

                prev_frame = None
                for fi, frame in enumerate(frames):
                    if count >= target:
                        break
                    h, w = frame.shape[:2]
                    if h < 480 or w < 640:
                        prev_frame = frame
                        continue

                    try:
                        depth_map = depth_est.predict(frame)
                        if prev_frame is not None:
                            flow = flow_est.predict(prev_frame, frame)
                        else:
                            flow = np.zeros((h, w, 2), dtype=np.float32)
                    except:
                        prev_frame = frame
                        continue

                    for ci in range(CROPS_PER_FRAME):
                        if count >= target:
                            break
                        # Better 3x balance: 40% chance of 3x
                        scale = 3 if rng.random() < 0.4 else 2
                        crop_gt = CROP_IN * scale
                        if h < crop_gt or w < crop_gt:
                            scale = 2
                            crop_gt = CROP_IN * 2
                        if h < crop_gt or w < crop_gt:
                            continue

                        cy = rng.randint(0, h - crop_gt)
                        cx = rng.randint(0, w - crop_gt)

                        gt_crop = frame[cy:cy+crop_gt, cx:cx+crop_gt].astype(np.float32) / 255.0
                        color_crop = cv2.resize(gt_crop, (CROP_IN, CROP_IN), interpolation=cv2.INTER_AREA)

                        from game_stylize import apply_game_style
                        if is_real:
                            color_crop = apply_game_style(color_crop, style="random", rng=np.random.default_rng())

                        depth_crop = cv2.resize(depth_map[cy:cy+crop_gt, cx:cx+crop_gt], (CROP_IN, CROP_IN))
                        flow_crop = cv2.resize(flow[cy:cy+crop_gt, cx:cx+crop_gt], (CROP_IN, CROP_IN))
                        flow_crop /= scale

                        sample = {
                            'color': torch.from_numpy(color_crop).permute(2, 0, 1).half(),
                            'depth': torch.from_numpy(depth_crop).unsqueeze(0).half(),
                            'motion_vectors': torch.from_numpy(flow_crop).permute(2, 0, 1).half(),
                            'ground_truth': torch.from_numpy(gt_crop).permute(2, 0, 1).half(),
                            'is_real': torch.tensor(is_real, dtype=torch.bool),
                            'scale': torch.tensor(scale, dtype=torch.int32),
                        }
                        save_sample_uint8(sample, output / f"sample_{out_idx:06d}.pt")
                        out_idx += 1
                        count += 1

                    prev_frame = frame
            except:
                continue

        total_generated += count
        print(f"  Generated {count} samples from {name}")

    # Final stats
    elapsed = time.time() - t_start
    all_files = list(output.glob("sample_*.pt"))
    real_count = 0
    synth_count = 0
    scale_2 = 0
    scale_3 = 0
    for f in random.sample(all_files, min(500, len(all_files))):
        s = torch.load(f, weights_only=False)
        if s['is_real'].item():
            real_count += 1
        else:
            synth_count += 1
        sc = s.get('scale', torch.tensor(2)).item()
        if sc == 2:
            scale_2 += 1
        else:
            scale_3 += 1

    sampled = real_count + synth_count
    print(f"\n{'='*50}")
    print(f"DONE: {total_generated} new + {len(all_files)} total samples")
    print(f"Sampled {sampled}: real={real_count} ({100*real_count/sampled:.0f}%) synth={synth_count} ({100*synth_count/sampled:.0f}%)")
    print(f"Scales: 2x={scale_2} ({100*scale_2/sampled:.0f}%) 3x={scale_3} ({100*scale_3/sampled:.0f}%)")
    print(f"Time: {elapsed/3600:.1f} hours")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
