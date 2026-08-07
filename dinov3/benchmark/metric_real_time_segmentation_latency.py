#!/usr/bin/env python3
"""
标准化实时语义分割延迟评测。评测结果不包含任何 CUDA/GPU 设备信息。
"""

from __future__ import annotations

from metric_common import (
    BENCHMARK_ITERS,
    BENCHMARK_STAGE,
    BENCHMARK_WARMUP,
    RAW_IMAGE_DIR,
    TARGET_LATENCY_MS,
    LATENCY_DIR,
    ensure_dir,
    build_m2f_runner,
    benchmark_runner,
    list_images,
    save_json,
)

RESULT_FILE = "real_time_segmentation_latency_result.json"
# 中期只需 1 张卡的单一延迟数; 终期沿用 GPU 0+2 双卡对比
TARGET_GPU_IDS = (0,) if BENCHMARK_STAGE == "midterm" else (0, 2)


def evaluate_on_device(device: str, files, run_label: str) -> dict:
    import torch

    gpu_idx = int(device.split(":")[1]) if ":" in device else 0

    print(f"\n[INFO] === {run_label} 评测中 ===")
    # 把当前线程默认 device 切到目标卡, 让 torch.cuda.synchronize() 等无参 API 落到正确卡上
    torch.cuda.set_device(gpu_idx)
    with torch.cuda.device(gpu_idx):
        infer_once, model_info = build_m2f_runner(files, 1, device, torch)
        stats = benchmark_runner(infer_once, 1, BENCHMARK_WARMUP, BENCHMARK_ITERS, device)

    record = {
        "label": run_label,
        **model_info,
        **stats,
        "target_latency_ms": TARGET_LATENCY_MS,
        "pass": stats["avg_frame_latency_ms"] <= TARGET_LATENCY_MS,
    }

    # 释放显存, 给下一次测量腾空间
    del infer_once
    torch.cuda.empty_cache()
    return record


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("本评测要求 CUDA")

    visible_count = torch.cuda.device_count()
    target_devices = [f"cuda:{idx}" for idx in TARGET_GPU_IDS if idx < visible_count]
    if not target_devices:
        raise RuntimeError("没有可用的目标设备")

    files = list_images(RAW_IMAGE_DIR)
    per_device_records = [
        evaluate_on_device(dev, files, f"run_{i + 1}")
        for i, dev in enumerate(target_devices)
    ]

    ensure_dir(LATENCY_DIR)
    out = LATENCY_DIR / RESULT_FILE

    if BENCHMARK_STAGE == "midterm":
        # 中期版: 单卡, 只输出延迟数值, 不暴露 GPU 型号 / p50 / p95 / fps
        rec = per_device_records[0]
        summary = {
            "metric": "real_time_segmentation_latency",
            "definition": "average per-frame latency from input to segmentation output",
            "target_latency_ms": TARGET_LATENCY_MS,
            "avg_frame_latency_ms": round(rec["avg_frame_latency_ms"], 2),
            "pass": rec["pass"],
        }
        save_json(out, summary)
        print()
        print(f"[INFO] result JSON: {out}")
        print()
        print(f"  avg_frame_latency_ms : {summary['avg_frame_latency_ms']:.2f} ms")
        print(f"  target_latency_ms    : {TARGET_LATENCY_MS:.1f} ms")
        print(f"  pass                 : {summary['pass']}")
        return 0

    # 终期版: 多次测量对比 + 完整字段 (不含任何 GPU/设备信息)
    summary = {
        "metric": "real_time_segmentation_latency",
        "definition": "average per-frame latency from input to segmentation output",
        "target_latency_ms": TARGET_LATENCY_MS,
        "runs": per_device_records,
        "best_run": min(per_device_records, key=lambda r: r["avg_frame_latency_ms"])["label"],
        "pass": any(r["pass"] for r in per_device_records),
    }
    save_json(out, summary)

    print()
    print(f"[INFO] result JSON: {out}")
    print()
    print(f"{'run':<10s} {'avg_frame_ms':>14s} {'p50_ms':>10s} {'p95_ms':>10s} {'fps':>8s}  pass")
    print("-" * 60)
    for r in per_device_records:
        print(
            f"{r['label']:<10s} {r['avg_frame_latency_ms']:>14.2f} "
            f"{r['p50_batch_latency_ms']:>10.2f} {r['p95_batch_latency_ms']:>10.2f} "
            f"{r['fps']:>8.2f}  {r['pass']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
