"""
ScanNet DINOv3 深度反投影 + 体素多帧投票 → 按类别输出 PCD

输入:
  - scannet_frames_25k/{scene}/depth/*.png        uint16 mm (640×480)
  - scannet_frames_25k/{scene}/pose/*.txt          4×4 camera-to-world
  - scannet_frames_25k/{scene}/intrinsics_depth.txt 4×4
  - scannet_frames_25k_dinov3_seg/{scene}/{frame}_mask_id.png  uint8 NYU40 (1296×968)
输出:
  - OUTPUT_ROOT/{scene}/{class_name}.pcd           按类别分离的语义点云
"""

import os
import sys
import time
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from collections import defaultdict

# ======================== 配置 ========================
SCANNET_ROOT = "/data1/data/scannet/scannet_frames_25k"
DINOV3_SEG_ROOT = "/data1/data/scannet/scannet_frames_25k_dinov3_seg"
OUTPUT_ROOT = "/data1/data/scannet/scannet_frames_25k_dinov3_seg/output/color_separated_scenes"

DEPTH_SCALE = 1000.0
MAX_DEPTH = 10.0
VOTE_VOXEL_SIZE = 0.02
VOTE_MIN_OBSERVATIONS = 3
NYU40_NUM_CLASSES = 41

SKIP_CLASSES = {0, 1, 2, 22}  # unlabeled, wall, floor, ceiling

NYU40_NAMES = [
    "unlabeled", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "blinds", "desk",
    "shelves", "curtain", "dresser", "pillow", "mirror", "floor_mat", "clothes",
    "ceiling", "books", "refrigerator", "television", "paper", "towel",
    "shower_curtain", "box", "whiteboard", "person", "night_stand", "toilet",
    "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop",
]


def load_intrinsic(path):
    return np.loadtxt(path)[:3, :3]


def load_pose(path):
    pose = np.loadtxt(path)
    if pose.shape != (4, 4):
        return None
    if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
        return None
    return pose


def load_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32) / DEPTH_SCALE


def backproject_depth(depth, intrinsic):
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                        np.arange(h, dtype=np.float32))

    valid = (depth > 0) & (depth < MAX_DEPTH)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=-1)
    pixel_indices = np.where(valid.ravel())[0]
    return points_cam, pixel_indices


def voxel_majority_vote(points, labels, n_classes, voxel_size=VOTE_VOXEL_SIZE,
                        min_obs=VOTE_MIN_OBSERVATIONS):
    voxel_ijk = np.floor(points / voxel_size).astype(np.int32)
    mins = voxel_ijk.min(axis=0)
    shifted = (voxel_ijk - mins).astype(np.int64)
    keys = (shifted[:, 0] << 40) | (shifted[:, 1] << 20) | shifted[:, 2]

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_voxels = len(unique_keys)

    vote_counts = np.zeros((n_voxels, n_classes), dtype=np.int32)
    np.add.at(vote_counts, (inverse, labels), 1)

    total_obs = vote_counts.sum(axis=1)
    winner = np.argmax(vote_counts, axis=1)

    valid = total_obs >= min_obs

    voxel_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(voxel_sum, inverse, points)
    voxel_count = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    voxel_centers = voxel_sum / np.maximum(voxel_count[:, None], 1)

    return voxel_centers[valid], winner[valid]


def process_scene(scene_name):
    scene_dir = os.path.join(SCANNET_ROOT, scene_name)
    seg_dir = os.path.join(DINOV3_SEG_ROOT, scene_name)
    out_dir = os.path.join(OUTPUT_ROOT, scene_name)

    if not os.path.isdir(scene_dir) or not os.path.isdir(seg_dir):
        return 0

    intrinsic_path = os.path.join(scene_dir, "intrinsics_depth.txt")
    if not os.path.exists(intrinsic_path):
        print(f"  跳过: intrinsics_depth.txt 不存在")
        return 0
    intrinsic = load_intrinsic(intrinsic_path)

    depth_dir = os.path.join(scene_dir, "depth")
    pose_dir = os.path.join(scene_dir, "pose")

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])
    if not depth_files:
        return 0

    all_points = []
    all_labels = []
    valid_count = 0

    for depth_file in depth_files:
        frame_id = os.path.splitext(depth_file)[0]

        depth_path = os.path.join(depth_dir, depth_file)
        pose_path = os.path.join(pose_dir, f"{frame_id}.txt")
        mask_path = os.path.join(seg_dir, f"{frame_id}_mask_id.png")

        if not os.path.exists(pose_path) or not os.path.exists(mask_path):
            continue

        depth = load_depth(depth_path)
        if depth is None:
            continue
        pose = load_pose(pose_path)
        if pose is None:
            continue

        mask_id = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_id is None:
            continue

        dh, dw = depth.shape
        if mask_id.shape[0] != dh or mask_id.shape[1] != dw:
            mask_id = cv2.resize(mask_id, (dw, dh), interpolation=cv2.INTER_NEAREST)

        points_cam, pix_idx = backproject_depth(depth, intrinsic)
        if len(points_cam) == 0:
            continue

        point_labels = mask_id.ravel()[pix_idx].astype(np.int32)

        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = points_cam @ R.T + t

        all_points.append(points_world)
        all_labels.append(point_labels)
        valid_count += 1

    if not all_points:
        return 0

    points_all = np.vstack(all_points)
    labels_all = np.concatenate(all_labels)

    points_voted, labels_voted = voxel_majority_vote(
        points_all, labels_all, NYU40_NUM_CLASSES
    )

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for cls_id in range(NYU40_NUM_CLASSES):
        if cls_id in SKIP_CLASSES:
            continue
        mask = labels_voted == cls_id
        if mask.sum() < 50:
            continue
        pts = points_voted[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cls_name = NYU40_NAMES[cls_id] if cls_id < len(NYU40_NAMES) else f"class_{cls_id}"
        o3d.io.write_point_cloud(os.path.join(out_dir, f"{cls_name}.pcd"), pcd)
        saved += 1

    return saved


def main():
    scene_dirs = sorted([
        d for d in os.listdir(SCANNET_ROOT)
        if os.path.isdir(os.path.join(SCANNET_ROOT, d, "depth"))
    ])
    print(f"[INFO] 共 {len(scene_dirs)} 个场景")
    print(f"[INFO] 输出目录: {OUTPUT_ROOT}")

    t_total = time.time()
    for i, scene_name in enumerate(scene_dirs):
        t0 = time.time()
        n_saved = process_scene(scene_name)
        elapsed = time.time() - t0
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(scene_dirs)}] {scene_name}: {n_saved} 类别, {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\n全部完成, 总耗时 {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
