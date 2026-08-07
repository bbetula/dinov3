# yolo_finetune — 门 / 障碍物 / 仪表检测架 三类 YOLOv8s 微调 pipeline

用 `edge_seg/video` 下的实拍视频，微调一个**闭集 YOLOv8s（3类）**检测器，
最终导出 ONNX，供 **isaac_ros_yolov8 / TensorRT** 在机狗 Jetson 上加速部署。

> **为什么是闭集 YOLOv8s 而不是开放词表**：实测 YOLO-World 零样本对这三类专业目标
> 基本检不出（门/障碍物/仪表架得分仅 0.03~0.07）。你的目标是「少数固定类别」，
> 闭集检测器天生更合适，微调后精度更稳，且有现成的 Isaac ROS 加速包。

## 环境

复用 `edge-seg` 环境（已有 ultralytics/torch/cv2）：
```bash
EP=/data1/user/miniconda3/envs/edge-seg/bin/python
# 导出 ONNX 时才需要：
$EP -m pip install onnx onnxslim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 目录

```
yolo_finetune/
├── ft_config.py          # 类别定义 / 路径 / 文件名→类别映射
├── 1_extract_frames.py   # 视频 → 帧（类别由文件名自动判定）
├── 2_build_dataset.py    # 组装 YOLO 数据集 + data.yaml（按视频时间切分）
├── 3_train.py            # 微调 YOLOv8s
├── 4_export_onnx.py      # 导出 ONNX（.plan 在 Jetson 上转）
├── frames/               # images/  labels/(你手动标)  classes.txt  manifest.json
├── dataset/              # (生成) images/{train,val} labels/{train,val} data.yaml
└── runs/                 # (生成) 训练输出
```

## 标注（手动）

标注**由你手动提供**。数据有个优势：每个视频只含一个已知类别（文件名带「门/障碍/仪表」），
所以标注时**不用分类，只需框位置**。

- 工具推荐 **[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)**（桌面，内置 SAM，点一下出紧框）
  或 labelImg。
- 打开 `frames/images/`，画框选类，**导出 YOLO 格式**到 `frames/labels/`
  （标签文件与图片同名 `.txt`）。
- 类别顺序**必须**与 `frames/classes.txt` 一致：`door=0, obstacle=1, instrument_rack=2`。

**YOLO 标签格式**：每张图一个同名 `.txt`，每行一个框 `class_id cx cy w h`
（中心坐标+宽高，全部按图宽高归一化到 0~1）。例：
```
1 0.703125 0.576389 0.125000 0.152778   # obstacle，中心(0.70,0.58) 宽高(0.125,0.153)
```
没有目标的图不需要标签文件（或空 txt）。X-AnyLabeling 导出时自动生成，无需手算。

## 一步步跑

```bash
EP=/data1/user/miniconda3/envs/edge-seg/bin/python
cd /data1/user/Dense-Object-level-Mapping/yolo_finetune

$EP 1_extract_frames.py --fps 4          # 抽帧（已抽好 193 帧在 frames/images/）
#   —— 用 X-AnyLabeling 手动标注 frames/images/ -> frames/labels/ ——
$EP 2_build_dataset.py                    # 组数据集（只纳入有标签的帧，按视频时间切分）
$EP 3_train.py --epochs 100              # 微调（先 --epochs 2 冒烟验证链路）
$EP 4_export_onnx.py --weights runs/yolov8s_3cls/weights/best.pt   # 导 ONNX
```

## 部署到机狗（isaac_ros_yolov8）

1. 把 `best.onnx` 拷到机狗 Jetson；
2. Jetson 上转引擎（引擎与硬件+TRT版本绑定，必须在 Jetson 上生成）：
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.plan --fp16
   ```
3. `isaac_ros_yolov8` 指向 `best.plan`，`num_classes=3`，类别名同 `CLASS_NAMES`。
   （3 类 ONNX 输出张量形状为 `[1, 7, 8400]`）

## 数据量提醒

当前 6 个视频（193 帧）只够**跑通流程 / 初步验证**，且每类基本只有 1 个物体实例、
1 个场景，模型会过拟合、换场景就废。要部署可用，需按实拍规范补数据：
每类 **200~300 张 / 300~500 个框**，覆盖**多个不同实例**、多场景、不同距离/角度/光照，
并留一个**没参与训练的场次**做真实评测。

## 训练结果与部署建议（首轮）

- 同分布 val：mAP50=0.995、mAP50-95=0.912（偏高，因 val 与 train 同视频，仅供参考）。
- 跨场景/跨相机（D455 机库）实测：料车/架子能识别（instrument_rack 0.7~0.84），
  但存在两个已知短板：
  1. **地面误检**：已通过加入 `frames/backgrounds/` 的纯空场景负样本大幅压制；
  2. **door ↔ instrument_rack 混淆**：两者都是银色金属框架，视觉相近 + 数据少所致，
     需更多样本区分。
- **背景负样本**：`frames/backgrounds/` 放纯空场景图（无任何目标），`2_build_dataset.py`
  会自动给它们写空标签加入训练，教模型「空地面=无目标」。建议从**真机部署环境**
  多取一些空场景帧放进去。
- **部署阈值**：真机 `isaac_ros_yolov8` 的置信度阈值建议设 **0.4~0.5**，进一步过滤低分误检
  （真目标通常 0.7+）。
