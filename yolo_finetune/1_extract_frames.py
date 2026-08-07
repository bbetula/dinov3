# -*- coding: utf-8 -*-
"""
Step 1：抽帧。把 edge_seg/video 下的每个视频按指定 fps 抽帧，
帧文件名用 ASCII（<类别>_v<视频号>__<帧号>.jpg，避免中文文件名给下游工具添麻烦），
并写 manifest.json 记录每帧的类别与来源视频。

用法：
  EP=/data1/user/miniconda3/envs/edge-seg/bin/python
  $EP 1_extract_frames.py                 # 默认 fps=4
  $EP 1_extract_frames.py --fps 6         # 抽密一点
"""
import os
import sys
import json
import glob
import argparse
import collections

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ft_config import (
    VIDEO_DIR, IMAGES_DIR, MANIFEST, FRAMES_DIR, class_of_video, CLASS_ZH,
)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=VIDEO_DIR)
    ap.add_argument("--fps", type=float, default=4.0, help="每秒抽多少帧")
    args = ap.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    videos = sorted(
        p for p in glob.glob(os.path.join(args.video_dir, "*"))
        if os.path.splitext(p)[1].lower() in VIDEO_EXTS
    )
    print(f"[extract] {len(videos)} 个视频 @ {args.fps} fps -> {IMAGES_DIR}")

    manifest = {}
    per_class_vid = collections.Counter()   # 每类的视频计数，用于生成 ASCII 视频号
    for vpath in videos:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        cls = class_of_video(stem)
        if cls is None:
            print(f"  [skip] 判不出类别：{stem}")
            continue
        per_class_vid[cls] += 1
        ascii_stem = f"{cls}_v{per_class_vid[cls]}"

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [warn] 打不开：{vpath}")
            continue
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(native_fps / args.fps))

        idx, saved = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                name = f"{ascii_stem}__{idx:06d}.jpg"
                cv2.imwrite(os.path.join(IMAGES_DIR, name), frame)
                manifest[name] = {"class": cls, "video": stem}
                saved += 1
            idx += 1
        cap.release()
        print(f"  {stem} [{CLASS_ZH[cls]}] -> {ascii_stem}: 抽 {saved} 帧")

    os.makedirs(FRAMES_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 汇总
    cnt = collections.Counter(v["class"] for v in manifest.values())
    print(f"\n[extract] 共 {len(manifest)} 帧，按类别：{dict(cnt)}")
    print(f"[extract] manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
