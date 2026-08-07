# -*- coding: utf-8 -*-
"""本地验证 best.onnx：取指定视频第一帧，跑推理，解码 [1,7,8400]，打印检测框。
预处理对齐 YOLOv8/Isaac ROS：letterbox->640, RGB, /255, NCHW float32。
"""
import cv2, numpy as np, onnxruntime as ort

ONNX = "/data1/user/Dense-Object-level-Mapping/yolo_finetune/runs/yolov8s_3cls_bg/weights/best.onnx"
VIDEO = "/data1/user/Dense-Object-level-Mapping/edge_seg/video/instrument_rack/instrument_rack_v4.mp4"
NAMES = ["door", "obstacle", "instrument_rack"]
IMGSZ = 640
CONF = 0.05          # 故意放低，看模型原始能力
IOU = 0.45


def letterbox(img, new=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    out = np.full((new, new, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def nms(boxes, scores, iou_thr):
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]; keep.append(i)
        if idx.size == 1: break
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[idx[1:], 2] - boxes[idx[1:], 0]) * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
        iou = inter / (a + b - inter + 1e-9)
        idx = idx[1:][iou <= iou_thr]
    return keep


def main():
    cap = cv2.VideoCapture(VIDEO)
    ok, frame = cap.read()
    cap.release()
    assert ok, "读不到第一帧"
    H, W = frame.shape[:2]
    print(f"[frame] 原始尺寸 {W}x{H}")

    lb, r, padx, pady = letterbox(frame, IMGSZ)
    blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]  # NCHW

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = sess.run(None, {iname: blob})[0]      # [1,7,8400]
    print(f"[onnx] 输出形状 {out.shape}")
    pred = out[0].T                              # [8400,7]
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]                     # 3 类
    cls_id = cls_scores.argmax(1)
    conf = cls_scores.max(1)

    m = conf > CONF
    boxes_xywh, cls_id, conf = boxes_xywh[m], cls_id[m], conf[m]
    print(f"[decode] 过 conf>{CONF} 的候选框: {len(conf)}")
    if len(conf) == 0:
        print(">>> 模型对这一帧没有任何候选，属于模型/域问题，不是部署配置问题")
        return

    # xywh(letterbox坐标) -> xyxy(原图)
    xy = boxes_xywh.copy()
    xy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    xy[:, [0, 2]] = (xy[:, [0, 2]] - padx) / r
    xy[:, [1, 3]] = (xy[:, [1, 3]] - pady) / r

    keep = nms(xy, conf, IOU)
    print(f"[nms] 最终框: {len(keep)}\n")
    frame_draw = frame.copy()
    for i in keep:
        x1, y1, x2, y2 = xy[i].astype(int)
        c, s = NAMES[cls_id[i]], conf[i]
        print(f"  {c:16s} conf={s:.3f}  box=({x1},{y1},{x2},{y2})")
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_draw, f"{c} {s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    outimg = "/data1/user/Dense-Object-level-Mapping/yolo_finetune/_onnx_infer_test.jpg"
    cv2.imwrite(outimg, frame_draw)
    print(f"\n[save] 可视化 -> {outimg}")


if __name__ == "__main__":
    main()
