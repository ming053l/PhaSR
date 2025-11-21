import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def get_normal_map_by_point_cloud(depth, fov):
    """Compute surface normal map from a depth map using point cloud."""
    height, width = depth.shape

    fov_radians = np.deg2rad(fov)
    focal_length = width / (2 * np.tan(fov_radians / 2))
    fx = focal_length
    fy = focal_length
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    def normalization(data):
        mo_chang = np.sqrt(
            np.multiply(data[:, :, 0], data[:, :, 0])
            + np.multiply(data[:, :, 1], data[:, :, 1])
            + np.multiply(data[:, :, 2], data[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return data / mo_chang

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))  # 3 x N

    pts_3d = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))  # 3 x N
    pts_3d_world = pts_3d.reshape((3, height, width))

    # forward (x) & top (y) neighbors to compute normal
    f = (
        pts_3d_world[:, 1:height - 1, 2:width]
        - pts_3d_world[:, 1:height - 1, 1:width - 1]
    )
    t = (
        pts_3d_world[:, 2:height, 1:width - 1]
        - pts_3d_world[:, 1:height - 1, 1:width - 1]
    )

    normal_map = -np.cross(f, t, axisa=0, axisb=0)  # H-2 x W-2 x 3
    normal_map = normalization(normal_map)

    new_normal_map = np.zeros((height, width, 3), dtype=normal_map.dtype)
    new_normal_map[1:height - 1, 1:width - 1, :] = normal_map
    return new_normal_map


def depthToPoint(fov, depth):
    """Convert depth map to 3D point cloud."""
    height, width = depth.shape
    fov_radians = np.deg2rad(fov)

    focal_length = width / (2 * np.tan(fov_radians / 2))
    fx = focal_length
    fy = focal_length
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    x, y = np.meshgrid(range(width), range(height))
    z = depth
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    x_3d = x_3d.astype(np.float32)
    y_3d = y_3d.astype(np.float32)

    point_cloud = np.stack((x_3d, y_3d, z), axis=-1)
    return point_cloud


def process_normal(normal):
    """Optional normal post-process (unused in your original loop)."""
    normal = normal * 2.0 - 1.0
    normal = normal[:, :, np.newaxis, :]
    normalizer = np.sqrt(normal @ normal.transpose(0, 1, 3, 2))
    normalizer = np.squeeze(normalizer, axis=-2)
    normalizer = np.clip(normalizer, 1.0e-20, 1.0e10)
    normal = np.squeeze(normal, axis=-2)
    normal = normal / normalizer
    return normal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate normal maps and visualizations from depth .npy files."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Root folder that contains 'depth' subfolder.",
    )
    parser.add_argument(
        "--depth-subdir",
        type=str,
        default="depth",
        help="Name of depth subdirectory inside folder.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera field-of-view (degrees).",
    )
    parser.add_argument(
        "--save-vis",
        default=True,
        help="Whether to save depth/normal visualization images.",
    )
    parser.add_argument(
        "--save-normal",
        default=True,
        help="Whether to save normal .npy files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    folder = args.source_root
    depth_folder = os.path.join(folder, args.depth_subdir)

    # Output dirs
    normal_dir = os.path.join(folder, "normal")
    depth_vis_dir = os.path.join(folder, "depth_vis")
    normal_vis_dir = os.path.join(folder, "normal_vis")

    if args.save_normal:
        os.makedirs(normal_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(depth_vis_dir, exist_ok=True)
        os.makedirs(normal_vis_dir, exist_ok=True)

    from PIL import Image

    for filename in tqdm(os.listdir(depth_folder), desc="Processing depth maps"):
        if not filename.endswith(".npy"):
            continue

        filename_1 = os.path.splitext(filename)[0]
        depth_path = os.path.join(depth_folder, filename)

        try:
            depth_map = np.load(depth_path)
        except FileNotFoundError:
            continue


        depth_map = cv2.medianBlur(depth_map, 5)
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0.0)
        depth_map = cv2.medianBlur(depth_map, 5)

        fov = args.fov


        pos_image_data = depthToPoint(fov, depth_map)
        normal_image_data = get_normal_map_by_point_cloud(depth_map, fov)
        normal_image_data = cv2.GaussianBlur(
            normal_image_data, (5, 5), 0.0
        ).astype(np.float32)


        pos_image_data[:, :, 2] = -pos_image_data[:, :, 2]
        normal_image_data[:, :, 2] = -normal_image_data[:, :, 2]


        normal_image_data = (normal_image_data + 1) * 0.5


        pos_image_data = pos_image_data[:, :, (2, 1, 0)]
        normal_image_data = normal_image_data[:, :, (2, 1, 0)]


        if args.save_normal:
            normal_image_data_npy = np.transpose(normal_image_data, (2, 0, 1))
            normal_path = os.path.join(normal_dir, filename)
            np.save(normal_path, normal_image_data_npy)


        if args.save_vis:
            depth_vis_path = os.path.join(depth_vis_dir, filename_1 + ".png")
            normal_vis_path = os.path.join(normal_vis_dir, filename_1 + ".png")


            plt.imsave(depth_vis_path, depth_map, cmap="viridis")


            scale_factor = 255
            normal_vis = (normal_image_data * scale_factor).astype(np.uint8)
            img = Image.fromarray(normal_vis)
            img.save(normal_vis_path)

        print(f"{filename} finish\n")


if __name__ == "__main__":
    main()
