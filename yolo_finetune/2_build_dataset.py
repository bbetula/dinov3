# -*- coding: utf-8 -*-
"""
Step 2：组装 YOLO 数据集。直接扫描 frames/images + frames/labels（不依赖 manifest），
只纳入「有非空标签」的帧。按视频源分组，**每段视频都均匀抽出一部分作 val**
（等间隔抽取，覆盖整段视频的视角，而非只取末尾），保证每个视频都在 val 中有代表。
若 frames/backgrounds/ 存在，其中的纯空场景图作为背景负样本（空标签）加入训练，
压制地面误检。生成 dataset/ 与 data.yaml。

用法：
  EP=/data1/user/miniconda3/envs/edge-seg/bin/python
  $EP 2_build_dataset.py                 # 默认 val_ratio=0.2
  $EP 2_build_dataset.py --val-ratio 0.15
"""
import os
import sys
import glob
import shutil
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ft_config import IMAGES_DIR, LABELS_DIR, DATASET_DIR, CLASS_NAMES, FRAMES_DIR

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def frame_index(name):
    try:
        return int(os.path.splitext(name)[0].split("__")[-1])
    except Exception:
        return 0


def video_key(name):
    return os.path.basename(name).split("__")[0]     # <cls>_v<n>


def spaced_val_indices(n, val_ratio):
    """在 0..n-1 里等间隔挑 round(n*val_ratio) 个作 val，覆盖整段视频。"""
    k = max(1, round(n * val_ratio)) if n > 1 else 0
    if k == 0:
        return set()
    return {round(i * (n - 1) / (k - 1)) if k > 1 else n // 2 for i in range(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.2, help="每段视频抽作 val 的比例")
    args = ap.parse_args()

    # 扫描有非空标签的帧
    labeled = []
    for p in sorted(glob.glob(os.path.join(IMAGES_DIR, "*"))):
        if os.path.splitext(p)[1].lower() not in IMG_EXTS:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(LABELS_DIR, stem + ".txt")
        if os.path.exists(lab) and os.path.getsize(lab) > 0:
            labeled.append(os.path.basename(p))

    if not labeled:
        print("⚠️ 没有任何已标注帧。请先用 X-AnyLabeling 人工标注后再运行。")
        return

    # 按视频分组，每段视频等间隔抽 val
    by_vid = collections.defaultdict(list)
    for name in labeled:
        by_vid[video_key(name)].append(name)

    splits = {"train": [], "val": []}
    per_vid_report = {}
    for vid, names in by_vid.items():
        names = sorted(names, key=frame_index)
        val_pos = spaced_val_indices(len(names), args.val_ratio)
        tr = va = 0
        for i, name in enumerate(names):
            if i in val_pos:
                splits["val"].append(name); va += 1
            else:
                splits["train"].append(name); tr += 1
        per_vid_report[vid] = (tr, va)

    # 落盘
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(DATASET_DIR, sub, sp), exist_ok=True)
    for sp, names in splits.items():
        for name in names:
            stem = os.path.splitext(name)[0]
            shutil.copy(os.path.join(IMAGES_DIR, name),
                        os.path.join(DATASET_DIR, "images", sp, name))
            shutil.copy(os.path.join(LABELS_DIR, stem + ".txt"),
                        os.path.join(DATASET_DIR, "labels", sp, stem + ".txt"))

    # ── 背景负样本（纯空场景，无目标）───────────────────────────────
    bg_dir = os.path.join(FRAMES_DIR, "backgrounds")
    n_bg = 0
    if os.path.isdir(bg_dir):
        for i, p in enumerate(sorted(glob.glob(os.path.join(bg_dir, "*")))):
            if os.path.splitext(p)[1].lower() not in IMG_EXTS:
                continue
            sp = "val" if (i % 7 == 6) else "train"     # ~85% train
            base = "bg_" + os.path.basename(p)
            stem = os.path.splitext(base)[0]
            shutil.copy(p, os.path.join(DATASET_DIR, "images", sp, base))
            open(os.path.join(DATASET_DIR, "labels", sp, stem + ".txt"), "w").close()
            n_bg += 1

    # data.yaml
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    # 统计
    def box_count(sp):
        c = collections.Counter()
        for l in glob.glob(os.path.join(DATASET_DIR, "labels", sp, "*.txt")):
            for line in open(l):
                if line.strip():
                    c[int(line.split()[0])] += 1
        return {CLASS_NAMES[k]: v for k, v in sorted(c.items())}

    print(f"[dataset] {len(by_vid)} 段视频；train {len(splits['train'])} / val {len(splits['val'])} 帧（含目标）")
    print(f"[dataset] 背景负样本（空标签）：{n_bg} 张")
    print(f"[dataset] train 框：{box_count('train')}")
    print(f"[dataset] val   框：{box_count('val')}")
    print(f"[dataset] 每段视频 train/val 帧数：")
    for vid in sorted(per_vid_report):
        tr, va = per_vid_report[vid]
        print(f"    {vid:22s} train {tr:3d} / val {va:2d}")
    print(f"[dataset] data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
