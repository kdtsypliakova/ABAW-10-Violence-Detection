#!/usr/bin/env python3
"""
precompute_features.py — Предвычисление skeleton и optical flow для DVD dataset.

Использование:
    # Skeleton (MediaPipe Pose, ~33 keypoints × 3 coords = 99 dim):
    python precompute_features.py --mode skeleton \
        --frames-root /path/to/frames \
        --output-dir /path/to/skeleton_features

    # Optical Flow (OpenCV Farneback, saved as 3ch images: dx, dy, magnitude):
    python precompute_features.py --mode flow \
        --frames-root /path/to/frames \
        --output-dir /path/to/flow_frames

Skeleton сохраняется как .npy: [n_frames, 406] float32 (top-2 people + velocities + pairwise features)
Flow сохраняется как .jpg кадры (так же как RGB frames, для совместимости с dataset).
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =====================================================================
#  Skeleton extraction via MediaPipe Tasks API (0.10.x)
# =====================================================================
def _ensure_pose_model(model_dir: Path) -> Path:
    """Download pose_landmarker model if not present."""
    model_path = model_dir / "pose_landmarker_lite.task"
    if model_path.exists():
        return model_path
    model_dir.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    print(f"Downloading pose model to {model_path} ...")
    import urllib.request
    urllib.request.urlretrieve(url, str(model_path))
    print("Done.")
    return model_path



def extract_skeleton_video(frames_dir: Path, output_path: Path, model_path: Path):
    """Извлекает позы из всех кадров директории, сохраняет в .npy [n_frames, 406]."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    frame_files = sorted(
        [f for f in frames_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')],
        key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or '0')
    )

    if len(frame_files) == 0:
        print(f"  [skip] no frames in {frames_dir}")
        detector.close()
        return False

    def person_vec(person_lms):
        out = []
        for lm in person_lms[:33]:
            out.extend([float(lm.x), float(lm.y), float(lm.z)])
        if len(out) < 99:
            out.extend([0.0] * (99 - len(out)))
        return np.asarray(out[:99], dtype=np.float32)

    def person_score(person_lms):
        vis = [float(getattr(lm, "visibility", 0.0)) for lm in person_lms[:33]]
        return float(np.mean(vis)) if vis else 0.0

    def pairwise_features(p1, p2):
        p1 = p1.reshape(33, 3)
        p2 = p2.reshape(33, 3)
        valid1 = np.any(p1 != 0.0, axis=1)
        valid2 = np.any(p2 != 0.0, axis=1)
        if valid1.any():
            c1 = p1[valid1, :2].mean(axis=0)
        else:
            c1 = np.zeros(2, dtype=np.float32)
        if valid2.any():
            c2 = p2[valid2, :2].mean(axis=0)
        else:
            c2 = np.zeros(2, dtype=np.float32)

        def d(i, j):
            if i >= 33 or j >= 33:
                return 0.0
            if not valid1[i] or not valid2[j]:
                return 0.0
            return float(np.linalg.norm(p1[i, :2] - p2[j, :2]))

        dist_centers = float(np.linalg.norm(c1 - c2))
        dist_mean = 0.0
        if valid1.any() and valid2.any():
            dist_mean = float(np.linalg.norm(p1[valid1, :2].mean(axis=0) - p2[valid2, :2].mean(axis=0)))

        feats = [
            dist_centers,
            d(15, 15), d(15, 16), d(16, 15), d(16, 16),  # wrists
            d(11, 11), d(12, 12),                         # shoulders
            d(23, 23), d(24, 24),                         # hips
            dist_mean,
        ]
        return np.asarray(feats, dtype=np.float32)

    all_feats = []
    prev_p1 = np.zeros(99, dtype=np.float32)
    prev_p2 = np.zeros(99, dtype=np.float32)

    for fpath in frame_files:
        img = cv2.imread(str(fpath))
        if img is None:
            p1 = np.zeros(99, dtype=np.float32)
            p2 = np.zeros(99, dtype=np.float32)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)

            people = []
            if result.pose_landmarks:
                for person in result.pose_landmarks[:2]:
                    people.append((person_score(person), person_vec(person)))
            people.sort(key=lambda x: x[0], reverse=True)

            p1 = people[0][1] if len(people) > 0 else np.zeros(99, dtype=np.float32)
            p2 = people[1][1] if len(people) > 1 else np.zeros(99, dtype=np.float32)

        v1 = p1 - prev_p1
        v2 = p2 - prev_p2
        pair = pairwise_features(p1, p2)
        feat = np.concatenate([p1, p2, v1, v2, pair], axis=0).astype(np.float32)  # 406
        all_feats.append(feat)
        prev_p1, prev_p2 = p1, p2

    detector.close()
    arr = np.stack(all_feats)  # [n_frames, 406]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return True


def run_skeleton(frames_root: Path, output_dir: Path):
    """Обрабатывает все видео."""
    # Download model once
    model_path = _ensure_pose_model(output_dir / ".model")

    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} videos in {frames_root}")

    for vdir in tqdm(video_dirs, desc="Skeleton extraction"):
        video_id = vdir.name
        out_path = output_dir / f"{video_id}.npy"
        if out_path.exists():
            continue
        try:
            extract_skeleton_video(vdir, out_path, model_path)
        except Exception as e:
            print(f"  [error] {video_id}: {e}")

    n_done = len(list(output_dir.glob("*.npy")))
    print(f"Skeleton extraction done: {n_done}/{len(video_dirs)} videos")


# =====================================================================
#  Optical Flow extraction via OpenCV Farneback
# =====================================================================
def extract_flow_video(frames_dir: Path, output_dir: Path, img_size: int = 224):
    """Вычисляет optical flow между соседними кадрами, сохраняет как 3ch images."""
    frame_files = sorted(
        [f for f in frames_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')],
        key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or '0')
    )

    if len(frame_files) < 2:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = None
    for i, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        if img is None:
            # Save zero flow
            zero = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            cv2.imwrite(str(output_dir / fpath.name), zero)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (img_size, img_size))

        if prev_gray is None:
            # First frame: zero flow
            zero = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            cv2.imwrite(str(output_dir / fpath.name), zero)
            prev_gray = gray
            continue

        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Convert to 3-channel image: dx, dy, magnitude
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx**2 + dy**2)

        # Normalize to 0-255
        def norm_channel(ch):
            mn, mx = ch.min(), ch.max()
            if mx - mn < 1e-6:
                return np.zeros_like(ch, dtype=np.uint8)
            return ((ch - mn) / (mx - mn) * 255).astype(np.uint8)

        flow_img = np.stack([norm_channel(dx), norm_channel(dy), norm_channel(mag)], axis=-1)
        cv2.imwrite(str(output_dir / fpath.name), flow_img)
        prev_gray = gray

    return True


def run_flow(frames_root: Path, output_dir: Path, img_size: int = 224):
    """Обрабатывает все видео."""
    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} videos in {frames_root}")

    for vdir in tqdm(video_dirs, desc="Optical flow"):
        video_id = vdir.name
        out_dir = output_dir / video_id
        if out_dir.exists() and len(list(out_dir.glob("*"))) > 0:
            continue
        try:
            extract_flow_video(vdir, out_dir, img_size)
        except Exception as e:
            print(f"  [error] {video_id}: {e}")

    n_done = len([d for d in output_dir.iterdir() if d.is_dir()])
    print(f"Optical flow done: {n_done}/{len(video_dirs)} videos")


# =====================================================================
#  Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['skeleton', 'flow'], required=True)
    ap.add_argument('--frames-root', required=True, help='Path to frames directory')
    ap.add_argument('--output-dir', required=True, help='Output directory')
    ap.add_argument('--img-size', type=int, default=224, help='Image size for flow')
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'skeleton':
        run_skeleton(frames_root, output_dir)
    elif args.mode == 'flow':
        run_flow(frames_root, output_dir, args.img_size)


if __name__ == '__main__':
    main()
