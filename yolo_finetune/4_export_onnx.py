# -*- coding: utf-8 -*-
"""
Step 5：导出 ONNX（供 isaac_ros_yolov8 / TensorRT 用）。

⚠️ 只导 ONNX；.plan（TensorRT 引擎）必须在目标 Jetson 上用 trtexec 生成，
   因为引擎与 GPU 架构 + TRT 版本绑定，不能在服务器生成再拷。

用法：
  EP=/data1/user/miniconda3/envs/edge-seg/bin/python
  # 先补装导出依赖（清华源）
  $EP -m pip install onnx onnxslim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
  # 导出
  $EP 4_export_onnx.py --weights runs/yolov8s_3cls/weights/best.pt

Jetson 上转引擎（在机狗上执行）：
  /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.plan --fp16
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ft_config import RUNS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(RUNS_DIR, "yolov8s_3cls", "weights", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=16)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"找不到权重：{args.weights}（先跑 3_train.py）")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)
    print(f"\n[export] ONNX -> {path}")
    print("[export] 输出张量形状 [1, 4+类别数, 8400]；本模型 3 类 -> [1, 7, 8400]")
    print("[export] 部署 isaac_ros_yolov8 时把 num_classes 配成 3。")
    print("[export] 在 Jetson 上转引擎：")
    print(f"         /usr/src/tensorrt/bin/trtexec --onnx={os.path.basename(path)} "
          f"--saveEngine={os.path.splitext(os.path.basename(path))[0]}.plan --fp16")


if __name__ == "__main__":
    main()
