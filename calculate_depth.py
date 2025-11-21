import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm.auto import tqdm
from depth_anything_v2.dpt import DepthAnythingV2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DepthAnythingV2 to generate depth maps (.npy)."
    )

    parser.add_argument(
        "--source-root",
        type=str,
        default="dataset",
        help="Path to the dataset root directory containing the origin folder.",
    )

    parser.add_argument(
        "--origin-subdir",
        type=str,
        default="origin",
        help="Name of the subdirectory containing original RGB images.",
    )

    parser.add_argument(
        "--depth-subdir",
        type=str,
        default="depth",
        help="Name of the output subdirectory for saving depth .npy files.",
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoints/depth_anything_v2_vitl.pth",
        help="Path to the DepthAnythingV2 checkpoint file.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID to use, e.g., '0', '1', or '0,1'.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Directories
    source_root = Path(args.source_root)
    origin_dir = source_root / args.origin_subdir
    depth_dir = source_root / args.depth_subdir
    depth_dir.mkdir(parents=True, exist_ok=True)

    if not origin_dir.exists():
        raise FileNotFoundError(f"Origin directory not found: {origin_dir}")

    # Load model
    print("📦 Loading DepthAnythingV2 model and checkpoint...")
    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    ).cuda()

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Collect input images
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = sorted([p for p in origin_dir.iterdir() if p.suffix.lower() in valid_ext])

    if len(files) == 0:
        print(f"⚠️ No valid images found in: {origin_dir}")
        return

    print("🚀 Starting depth inference...")
    progress = tqdm(files, desc="Depth Inference", unit="img")

    error_log = []

    # Run inference
    with torch.inference_mode():
        for image_path in progress:
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError("cv2.imread failed or image file corrupted")

                # Depth inference
                depth = model.infer_image(img)

                # Normalize to 0–255
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth = (depth * 255.0).astype(np.uint8)

                # Save depth as .npy
                np.save(depth_dir / f"{image_path.stem}.npy", depth)

            except Exception as e:
                msg = f"{image_path}: {str(e)}"
                error_log.append(msg)
                print("⚠️", msg)
                continue

    print("✅ Depth inference completed.")

    if error_log:
        print("\n⚠️ Errors encountered during processing:")
        for msg in error_log:
            print(" -", msg)


if __name__ == "__main__":
    main()
