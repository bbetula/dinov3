# -*- coding: utf-8 -*-
"""
yolo_finetune 配置：3 类目标（门 / 障碍物 / 仪表检测架）YOLOv8s 微调 pipeline

数据来源：edge_seg/video 下的实拍视频。每个视频只含一个已知类别，
文件名里带中文关键词，据此自动给该视频所有帧打上类别（标注时只需框位置）。
"""
import os

FT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FT_DIR)
EDGE_SEG_DIR = os.path.join(REPO_ROOT, "edge_seg")

# 默认输入视频目录
VIDEO_DIR = os.path.join(EDGE_SEG_DIR, "video")

# 微调基座（ultralytics 会自动下载）
BASE_YOLO = "yolov8s.pt"

# ── 类别定义（与 edge_seg 保持一致）──────────────────────────────────
CLASS_NAMES = ["door", "obstacle", "instrument_rack"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
CLASS_ZH = {"door": "门", "obstacle": "障碍物", "instrument_rack": "仪表检测架"}

# 视频文件名关键词 → 类别（用于自动给每个视频判类）
VIDEO_KEYWORD_TO_CLASS = {
    "门": "door",
    "障碍": "obstacle",
    "仪表": "instrument_rack",
}

# ── pipeline 目录 ────────────────────────────────────────────────────
FRAMES_DIR = os.path.join(FT_DIR, "frames")          # 抽帧输出
IMAGES_DIR = os.path.join(FRAMES_DIR, "images")      # 帧图
LABELS_DIR = os.path.join(FRAMES_DIR, "labels")      # YOLO 标签（人工标注）
MANIFEST = os.path.join(FRAMES_DIR, "manifest.json")  # 帧 → 类别/来源视频

DATASET_DIR = os.path.join(FT_DIR, "dataset")        # 训练用数据集（组装后）
RUNS_DIR = os.path.join(FT_DIR, "runs")              # 训练输出


def class_of_video(name: str):
    """按文件名关键词判断视频类别；判不出返回 None。"""
    for kw, cls in VIDEO_KEYWORD_TO_CLASS.items():
        if kw in name:
            return cls
    return None
