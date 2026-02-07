import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Depth-Anything-V2'))

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

def parse_args():
    parser = argparse.ArgumentParser(description='Depth and Normal Map Generation')
    parser.add_argument('--root', type=str, required=True, help='Dataset root path containing "origin" folder, use absolute path')
    parser.add_argument('--checkpoint', type=str, 
                        default="Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth",
                        help='Path to DepthAnythingV2 checkpoint')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Encoder type (vits, vitb, vitl, vitg)')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    return parser.parse_args()

def get_model_config(encoder):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    if encoder not in model_configs:
        raise ValueError(f"Unknown encoder: {encoder}")
    return model_configs[encoder]

def calculate_normal_map(img_path: Path, ksize=5):
    depth = np.load(img_path).astype(np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=ksize)
    dz = np.ones_like(dx) * -1
    normals = np.stack((dx, dy, dz), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= (norm + 1e-6)
    normal_map = (normals + 1) / 2 * 255
    return normal_map.astype("uint8").transpose(2, 0, 1)  # (C, H, W)

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Path settings
    source_root = Path(args.root)
    origin = source_root / "origin"
    depth_root = source_root / "depth"
    normal_root = source_root / "normal"
    
    if not origin.exists():
        raise FileNotFoundError(f"Origin folder not found at {origin}")
        
    depth_root.mkdir(parents=True, exist_ok=True)
    normal_root.mkdir(parents=True, exist_ok=True)
    
    # Model loading
    print(f"Loading model ({args.encoder}) from {args.checkpoint}...")
    config = get_model_config(args.encoder)
    model = DepthAnythingV2(**config).cuda()
    
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    
    # Depth inference phase
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    error_log = []
    
    files = [p for p in origin.iterdir() if p.suffix.lower() in valid_ext]
    bar = tqdm(files, desc="Depth Inference", unit="img")
    
    with torch.inference_mode():
        for image_path in bar:
            try:
                raw_img = cv2.imread(str(image_path))
                if raw_img is None:
                    raise ValueError("cv2.imread failed or file is corrupted")
                
                depth = model.infer_image(raw_img)
                # Normalize to 0-255
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                
                np.save(depth_root / f"{image_path.stem}.npy", depth)
                
            except Exception as e:
                error_msg = f"{image_path}: {str(e)}"
                error_log.append(error_msg)
                continue
    
    depth_files = list(depth_root.glob("*.npy"))
    bar = tqdm(depth_files, desc="Normal Map", unit="depth")
    
    for depth_img_path in bar:
        try:
            normal_map = calculate_normal_map(depth_img_path)
            np.save(normal_root / depth_img_path.name, normal_map)
        except Exception as e:
            error_msg = f"{depth_img_path}: {str(e)}"
            error_log.append(error_msg)
            continue
            
    # Error logging
    if error_log:
        with open(source_root / "error_log.txt", "w") as f:
            f.write("\n".join(error_log))



if __name__ == "__main__":
    main()
