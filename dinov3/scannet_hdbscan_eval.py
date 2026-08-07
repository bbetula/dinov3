"""
ScanNet HDBSCAN 实例聚类 → pred_bboxes JSON

输入:
  - color_separated_scenes/{scene}/{class_name}.pcd  (来自 scannet_depth_projection.py)
输出:
  - pred_bboxes/{scene}_pred_bboxes.json
"""

import os
import sys
import json
import time
import numpy as np
import open3d as o3d
import hdbscan
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# ======================== 配置 ========================
INPUT_ROOT = "/data1/data/scannet/scannet_frames_25k_dinov3_seg/output/color_separated_scenes"
OUTPUT_ROOT = "/data1/data/scannet/scannet_frames_25k_dinov3_seg/output/pred_bboxes"

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 3
MIN_CLUSTER_POINTS = 10
MAX_CLUSTER_FILTER = 50
NOISE_ABSORB_FACTOR = 15.0
AUTO_EPSILON = True
CLUSTER_SELECTION_METHOD = "eom"
MIN_PCD_POINTS = 50

SKIP_CLASSES = {"unlabeled", "wall", "floor", "ceiling"}

NYU40_NAME_TO_ID = {
    "unlabeled": 0, "wall": 1, "floor": 2, "cabinet": 3, "bed": 4,
    "chair": 5, "sofa": 6, "table": 7, "door": 8, "window": 9,
    "bookshelf": 10, "picture": 11, "counter": 12, "blinds": 13, "desk": 14,
    "shelves": 15, "curtain": 16, "dresser": 17, "pillow": 18, "mirror": 19,
    "floor_mat": 20, "clothes": 21, "ceiling": 22, "books": 23,
    "refrigerator": 24, "television": 25, "paper": 26, "towel": 27,
    "shower_curtain": 28, "box": 29, "whiteboard": 30, "person": 31,
    "night_stand": 32, "toilet": 33, "sink": 34, "lamp": 35, "bathtub": 36,
    "bag": 37, "otherstructure": 38, "otherfurniture": 39, "otherprop": 40,
}


def estimate_cluster_epsilon(points, k=5, sample_size=5000):
    if len(points) == 0:
        return 0.0
    if len(points) > sample_size:
        indices = np.random.choice(len(points), size=sample_size, replace=False)
        sample = points[indices]
    else:
        sample = points
    n_neighbors = min(k + 1, len(sample))
    if n_neighbors <= 1:
        return 0.1
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
    nn.fit(sample)
    distances, _ = nn.kneighbors(sample)
    knn_dists = np.sort(distances[:, -1])
    log_dists = np.log1p(knn_dists)
    gaps = np.diff(log_dists)
    gap_idx = int(np.argmax(gaps))

    if gaps[gap_idx] >= 2.0:
        epsilon = float(knn_dists[gap_idx])
    else:
        epsilon = float(np.percentile(knn_dists, 50))
    return float(np.clip(epsilon, 0.05, 2.0))


def filter_clusters_by_loggap(cluster_labels, min_points=MIN_CLUSTER_POINTS):
    unique, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
    if len(unique) <= 1:
        return cluster_labels

    counts_sorted_idx = np.argsort(counts)
    counts_sorted = counts[counts_sorted_idx]
    log_counts = np.log1p(counts_sorted.astype(float))
    gaps = np.diff(log_counts)
    gap_idx = int(np.argmax(gaps))

    if gaps[gap_idx] >= 1.0:
        threshold = int(counts_sorted[gap_idx]) + 1
    else:
        threshold = min_points
    threshold = max(min_points, min(threshold, MAX_CLUSTER_FILTER))

    small_cluster_ids = unique[counts < threshold]
    filtered = cluster_labels.copy()
    for cid in small_cluster_ids:
        filtered[filtered == cid] = -1
    return filtered


def absorb_noise_points(points, cluster_labels, epsilon, factor=NOISE_ABSORB_FACTOR):
    absorb_radius = factor * epsilon
    noise_mask = cluster_labels == -1
    if not np.any(noise_mask):
        return cluster_labels
    cluster_mask = cluster_labels != -1
    if not np.any(cluster_mask):
        return cluster_labels

    cluster_pts = points[cluster_mask]
    cluster_ids = cluster_labels[cluster_mask]
    noise_pts = points[noise_mask]

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(cluster_pts)
    distances, indices = nn.kneighbors(noise_pts)
    distances = distances[:, 0]
    indices = indices[:, 0]

    updated = cluster_labels.copy()
    noise_indices = np.where(noise_mask)[0]
    absorb_mask = distances <= absorb_radius
    updated[noise_indices[absorb_mask]] = cluster_ids[indices[absorb_mask]]
    return updated


def cluster_and_bbox(points, class_name):
    if len(points) < MIN_PCD_POINTS:
        return []

    if AUTO_EPSILON:
        epsilon = estimate_cluster_epsilon(points)
    else:
        epsilon = 0.0

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        gen_min_span_tree=False,
        core_dist_n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(points)
    cluster_labels = filter_clusters_by_loggap(cluster_labels)
    cluster_labels = absorb_noise_points(points, cluster_labels, epsilon)

    unique_clusters = [cid for cid in np.unique(cluster_labels) if cid != -1]
    bboxes = []
    for cid in unique_clusters:
        idxs = np.where(cluster_labels == cid)[0]
        cluster_pts = points[idxs]
        bbox_min = cluster_pts.min(axis=0).tolist()
        bbox_max = cluster_pts.max(axis=0).tolist()
        bbox_center = ((cluster_pts.min(axis=0) + cluster_pts.max(axis=0)) / 2).tolist()
        bbox_size = (cluster_pts.max(axis=0) - cluster_pts.min(axis=0)).tolist()
        bboxes.append({
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_center": bbox_center,
            "bbox_size": bbox_size,
            "num_points": int(len(idxs)),
            "nyu40class": class_name,
        })
    return bboxes


def process_scene(scene_name):
    scene_dir = os.path.join(INPUT_ROOT, scene_name)
    if not os.path.isdir(scene_dir):
        return []

    pcd_files = sorted([f for f in os.listdir(scene_dir) if f.endswith(".pcd")])
    all_bboxes = []

    for pcd_file in pcd_files:
        class_name = os.path.splitext(pcd_file)[0]
        if class_name in SKIP_CLASSES:
            continue

        pcd = o3d.io.read_point_cloud(os.path.join(scene_dir, pcd_file))
        if pcd.is_empty():
            continue
        points = np.asarray(pcd.points)
        bboxes = cluster_and_bbox(points, class_name)
        all_bboxes.extend(bboxes)

    return all_bboxes


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    scene_dirs = sorted([
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ])
    print(f"[INFO] 共 {len(scene_dirs)} 个场景")
    print(f"[INFO] 输出目录: {OUTPUT_ROOT}")

    t_total = time.time()
    total_bboxes = 0

    for i, scene_name in enumerate(scene_dirs):
        t0 = time.time()
        bboxes = process_scene(scene_name)
        total_bboxes += len(bboxes)

        out_path = os.path.join(OUTPUT_ROOT, f"{scene_name}_pred_bboxes.json")
        with open(out_path, "w") as f:
            json.dump(bboxes, f, indent=2)

        elapsed = time.time() - t0
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(scene_dirs)}] {scene_name}: {len(bboxes)} bboxes, {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\n全部完成: {len(scene_dirs)} 场景, {total_bboxes} bboxes, "
          f"总耗时 {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
