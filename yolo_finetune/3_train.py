# -*- coding: utf-8 -*-
"""
Step 3：微调 YOLOv8s（3 类）。针对本数据集的实际情况（样本少、单实例、obstacle 为
小目标、部署环境偏暗）做了强化配置，目标是最大化对标注类别的检测能力：

  · 强增强抗过拟合：mosaic + mixup + 多尺度 + 随机擦除
  · 尺度/亮度增强贴合部署：scale 抖动（多距离）、hsv_v 提高（暗光）
  · 余弦退火 + 早停 + 后段关闭 mosaic 稳定收敛

用法：
  EP=/data1/user/miniconda3/envs/edge-seg/bin/python
  $EP 3_train.py                     # 用下面的优化默认配置
  $EP 3_train.py --epochs 2          # 快速冒烟验证链路
  $EP 3_train.py --model yolov8m.pt  # 想试更大模型（数据多了再考虑）
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ft_config import DATASET_DIR, RUNS_DIR, BASE_YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATASET_DIR, "data.yaml"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=BASE_YOLO, help="基座权重或续训的 best.pt")
    ap.add_argument("--name", default="yolov8s_3cls")
    ap.add_argument("--device", default="0")
    ap.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（0=全微调）")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=RUNS_DIR,
        name=args.name,
        exist_ok=True,
        # ── 收敛 / 正则（小数据）───────────────────────────
        optimizer="auto",
        cos_lr=True,                 # 余弦退火，末期精修
        patience=max(40, args.epochs // 4),   # 早停
        warmup_epochs=3.0,
        weight_decay=0.0005,
        freeze=args.freeze or None,
        # ── 几何增强（多距离 / 多视角鲁棒）─────────────────
        degrees=8.0,                 # 小角度旋转
        translate=0.1,
        scale=0.5,                   # 尺度抖动：覆盖远近不同大小（对小目标 obstacle 关键）
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,
        # ── 光度增强（贴合部署暗光 / 机库杂光）──────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,                   # 提高亮度扰动，抗暗光域偏移
        # ── 合成增强（小数据抗过拟合的主力）────────────────
        mosaic=1.0,
        close_mosaic=20,             # 最后 20 轮关闭 mosaic，稳定框回归
        mixup=0.15,
        erasing=0.4,                 # 随机擦除，抗遮挡 + 正则
        multi_scale=True,            # 多尺度训练，增强尺度鲁棒
        # ── 输出 ───────────────────────────────────────────
        plots=True,
        verbose=True,
    )

    best = os.path.join(RUNS_DIR, args.name, "weights", "best.pt")
    print(f"\n[train] 完成。最佳权重 -> {best}")

    # 训练后单独跑一次验证，打印每类指标
    print("\n[train] ===== 验证集每类指标 =====")
    metrics = YOLO(best).val(data=args.data, imgsz=args.imgsz, device=args.device,
                             project=RUNS_DIR, name=args.name + "_val", exist_ok=True)
    try:
        names = metrics.names
        print(f"  整体  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  "
              f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}")
        for i, c in enumerate(metrics.box.ap_class_index):
            print(f"  {names[c]:16s} AP50={metrics.box.ap50[i]:.3f}  "
                  f"AP50-95={metrics.box.ap[i]:.3f}")
    except Exception as e:
        print("  (指标解析跳过:", e, ")")

    print(f"\n[train] 下一步导出 ONNX：$EP 4_export_onnx.py --weights {best}")


if __name__ == "__main__":
    main()
