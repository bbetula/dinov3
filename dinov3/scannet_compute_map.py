"""
ScanNet GT bbox 提取 + mAP@0.25/0.50 计算

GT bbox: 从 scannet_frames_25k/{scene}/instance/*.png + depth + pose 反投影
Pred bbox: 从 pred_bboxes/{scene}_pred_bboxes.json 读取

输出: map_results.txt (与 scans_workspace/output/map_results.txt 格式一致)
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from collections import defaultdict

# ======================== 配置 ========================
SCANNET_ROOT = "/data1/data/scannet/scannet_frames_25k"
PRED_BBOX_DIR = "/data1/data/scannet/scannet_frames_25k_dinov3_seg/output/pred_bboxes"
OUTPUT_DIR = "/data1/data/scannet/scannet_frames_25k_dinov3_seg/output"

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

NYU40_NAME_TO_ID = {name: i for i, name in enumerate(NYU40_NAMES)}

BENCHMARK_18_IDS = {36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34, 6, 7, 33, 9}
BENCHMARK_18_NAMES = {
    "bathtub", "bed", "bookshelf", "cabinet", "chair", "counter", "curtain",
    "desk", "door", "otherfurniture", "picture", "refrigerator", "shower_curtain",
    "sink", "sofa", "table", "toilet", "window",
}

MIN_GT_POINTS = 50


# ======================== 深度反投影 (复用) ========================
def load_intrinsic(path):
    return np.loadtxt(path)[:3, :3]


def load_pose(path):
    pose = np.loadtxt(path)
    if pose.shape != (4, 4):
        return None
    if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
        return None
    return pose


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


# ======================== GT bbox 提取 ========================
def extract_gt_bboxes(scene_name):
    scene_dir = os.path.join(SCANNET_ROOT, scene_name)
    depth_dir = os.path.join(scene_dir, "depth")
    pose_dir = os.path.join(scene_dir, "pose")
    instance_dir = os.path.join(scene_dir, "instance")
    intrinsic_path = os.path.join(scene_dir, "intrinsics_depth.txt")

    if not os.path.isdir(instance_dir) or not os.path.exists(intrinsic_path):
        return []

    intrinsic = load_intrinsic(intrinsic_path)
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    instance_points = defaultdict(list)

    for depth_file in depth_files:
        frame_id = os.path.splitext(depth_file)[0]
        depth_path = os.path.join(depth_dir, depth_file)
        pose_path = os.path.join(pose_dir, f"{frame_id}.txt")
        inst_path = os.path.join(instance_dir, f"{frame_id}.png")

        if not os.path.exists(pose_path) or not os.path.exists(inst_path):
            continue

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / DEPTH_SCALE

        pose = load_pose(pose_path)
        if pose is None:
            continue

        inst_map = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
        if inst_map is None:
            continue

        dh, dw = depth.shape
        if inst_map.shape[0] != dh or inst_map.shape[1] != dw:
            inst_map = cv2.resize(inst_map, (dw, dh), interpolation=cv2.INTER_NEAREST)

        points_cam, pix_idx = backproject_depth(depth, intrinsic)
        if len(points_cam) == 0:
            continue

        inst_labels = inst_map.ravel()[pix_idx].astype(np.int32)

        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = points_cam @ R.T + t

        for inst_id in np.unique(inst_labels):
            if inst_id == 0:
                continue
            class_id = inst_id // 1000
            if class_id in SKIP_CLASSES:
                continue
            mask = inst_labels == inst_id
            instance_points[inst_id].append(points_world[mask])

    gt_bboxes = []
    for inst_id, pts_list in instance_points.items():
        pts = np.vstack(pts_list)
        if len(pts) < MIN_GT_POINTS:
            continue
        class_id = inst_id // 1000
        if class_id >= len(NYU40_NAMES):
            continue
        class_name = NYU40_NAMES[class_id]
        bbox_min = pts.min(axis=0).tolist()
        bbox_max = pts.max(axis=0).tolist()
        gt_bboxes.append({
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "nyu40class": class_name,
            "nyu40id": int(class_id),
            "instance_id": int(inst_id),
            "num_points": int(len(pts)),
        })

    return gt_bboxes


# ======================== 3D IoU 计算 ========================
def compute_3d_iou(box_a, box_b):
    a_min = np.array(box_a["bbox_min"])
    a_max = np.array(box_a["bbox_max"])
    b_min = np.array(box_b["bbox_min"])
    b_max = np.array(box_b["bbox_max"])

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter_size[0] * inter_size[1] * inter_size[2]

    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = a_size[0] * a_size[1] * a_size[2]
    b_vol = b_size[0] * b_size[1] * b_size[2]

    union_vol = a_vol + b_vol - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


# ======================== AP 计算 ========================
def compute_ap(pred_bboxes, gt_bboxes, iou_threshold):
    if not gt_bboxes:
        return 0.0, 0, len(pred_bboxes), 0

    pred_sorted = sorted(pred_bboxes, key=lambda x: x["num_points"], reverse=True)

    gt_matched = [False] * len(gt_bboxes)
    tp_list = []
    fp_list = []
    tp_count = 0

    for pred in pred_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_bboxes):
            if gt_matched[gi]:
                continue
            iou = compute_3d_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            tp_list.append(1)
            fp_list.append(0)
            tp_count += 1
        else:
            tp_list.append(0)
            fp_list.append(1)

    if not tp_list:
        return 0.0, len(gt_bboxes), 0, 0

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    recalls = tp_cumsum / len(gt_bboxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # all-point interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap, len(gt_bboxes), len(pred_sorted), tp_count


# ======================== 主评测函数 ========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scene_dirs = sorted([
        d for d in os.listdir(SCANNET_ROOT)
        if os.path.isdir(os.path.join(SCANNET_ROOT, d, "depth"))
    ])
    print(f"[INFO] 共 {len(scene_dirs)} 个场景")

    all_pred_by_class = defaultdict(list)
    all_gt_by_class = defaultdict(list)
    scenes_with_pred = 0
    scenes_with_gt = 0

    t_total = time.time()
    for i, scene_name in enumerate(scene_dirs):
        t0 = time.time()

        # GT
        gt_bboxes = extract_gt_bboxes(scene_name)
        if gt_bboxes:
            scenes_with_gt += 1
        for gt in gt_bboxes:
            all_gt_by_class[gt["nyu40class"]].append(gt)

        # Pred
        pred_path = os.path.join(PRED_BBOX_DIR, f"{scene_name}_pred_bboxes.json")
        pred_bboxes = []
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                pred_bboxes = json.load(f)
            if pred_bboxes:
                scenes_with_pred += 1
        for pred in pred_bboxes:
            all_pred_by_class[pred["nyu40class"]].append(pred)

        elapsed = time.time() - t0
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[{i+1}/{len(scene_dirs)}] GT: {len(gt_bboxes)}, Pred: {len(pred_bboxes)}, {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\nGT/Pred 收集完成, 总耗时 {total_time:.1f}s")

    all_classes = sorted(set(list(all_gt_by_class.keys()) + list(all_pred_by_class.keys())))
    all_classes_37 = [c for c in all_classes if NYU40_NAME_TO_ID.get(c, 0) not in SKIP_CLASSES]
    all_classes_18 = [c for c in all_classes_37 if c in BENCHMARK_18_NAMES]

    pred_classes = set(c for c in all_pred_by_class if len(all_pred_by_class[c]) > 0)
    gt_classes = set(c for c in all_gt_by_class if len(all_gt_by_class[c]) > 0)

    out_path = os.path.join(OUTPUT_DIR, "map_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Pred scenes: {scenes_with_pred}  |  GT scenes: {scenes_with_gt}  |  "
                f"Matched: {min(scenes_with_pred, scenes_with_gt)}  |  "
                f"GT-only (no pred): {max(0, scenes_with_gt - scenes_with_pred)}\n")
        f.write(f"Pred classes: {len(pred_classes)}  |  GT classes: {len(gt_classes)}\n\n")

        for mode_name, class_list in [
            ("Mode 1: NYU40 minus wall/floor/ceiling/unlabeled (37 classes)", all_classes_37),
            ("Mode 2: ScanNet 18 benchmark", all_classes_18),
        ]:
            f.write(f"{'='*60}\n{mode_name}\n{'='*60}\n\n")

            for iou_thresh in [0.25, 0.50]:
                aps = []
                lines = []
                for cls_name in class_list:
                    pred_cls = all_pred_by_class.get(cls_name, [])
                    gt_cls = all_gt_by_class.get(cls_name, [])
                    ap, n_gt, n_pred, n_tp = compute_ap(pred_cls, gt_cls, iou_thresh)
                    aps.append(ap)
                    lines.append(f"  {cls_name:<28} {ap*100:6.2f}%  {n_gt:>6}  {n_pred:>6}  {n_tp:>5}")

                mAP = np.mean(aps) * 100 if aps else 0.0
                f.write(f"  mAP@{iou_thresh}: {mAP:.2f}\n")
                f.write(f"  {'Class':<28} {'AP':>7}  {'#GT':>6}  {'#Pred':>6}  {'#TP':>5}\n")
                f.write(f"  {'-'*57}\n")
                for line in lines:
                    f.write(line + "\n")
                f.write(f"\n")

                print(f"  {mode_name[:20]}... mAP@{iou_thresh}: {mAP:.2f}")

    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
