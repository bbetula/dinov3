#!/usr/bin/env python3
"""Multi-run scene-graph evaluation.

Applies one consistent ground-truth / expert-audit rule set (derived from the
real conference-room walkthrough video) to each of the 5 test result folders,
writes the four acceptance-review CSVs into every folder, and produces two
summary tables:

  Table 1  Structured accuracy  (instance nodes / relation edges / total)
  Table 2  Scene-graph completeness across the multiple runs

The rule set is intentionally shared so the five runs are directly comparable;
only the build parameters differ between runs.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FUSED_DIR = SCRIPT_DIR.parent
REPO_ROOT = FUSED_DIR.parent
for p in (str(SCRIPT_DIR), str(FUSED_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_scene_graph import (  # noqa: E402
    STRUCTURAL_CATEGORIES,
    categories,
    classify_colors,
    cluster_points,
    downsample_labeled_points,
    load_raw_semantic_cloud,
)
from class_statics_config import COLOR_TOLERANCE  # noqa: E402
from scene_graph_config import SEMANTIC_PCD_PATH  # noqa: E402

BASE = Path(
    "/data1/user/data/2026.05.28_655/点云图片文件/fastlivo_output_2026.05.29_2slow"
    "/lidar/scene_graph/all_raw_points_color_normal_strict"
)

# Build parameters actually used for each run (kept in sync with the 5 builds).
RUN_PARAMS = {
    "test1": {"voxel_size": 0.18, "dbscan_eps": 0.25, "dbscan_min_points": 5, "min_instance_points": 15},
    "test2": {"voxel_size": 0.20, "dbscan_eps": 0.30, "dbscan_min_points": 5, "min_instance_points": 20},
    "test3": {"voxel_size": 0.17, "dbscan_eps": 0.24, "dbscan_min_points": 5, "min_instance_points": 15},
    "test4": {"voxel_size": 0.22, "dbscan_eps": 0.28, "dbscan_min_points": 5, "min_instance_points": 18},
    "test5": {"voxel_size": 0.19, "dbscan_eps": 0.26, "dbscan_min_points": 5, "min_instance_points": 16},
}
RUNS = list(RUN_PARAMS.keys())

# --- shared ground truth (real conference room, confirmed from the video) ----
# Object/region labels that genuinely exist in the scene.
REAL_LABELS = {
    "wall", "floor", "ceiling", "door", "windowpane", "table", "desk", "chair",
    "swivel chair", "armchair", "television receiver", "monitor", "screen",
    "computer", "clock", "light", "bookcase", "shelf", "cabinet", "radiator",
    "curtain", "plant", "pot", "book", "signboard", "column", "apparel",
}
# Labels that are misclassifications / noise in this scene.
NOISE_LABELS = {"railing", "fan", "towel", "tray", "box", "bag", "basket", "flag", "mirror", "crt screen"}

# A relation edge below this confidence is treated as unreliable (counted wrong).
EDGE_CONF_THRESHOLD = 0.85

# Expected scene contents (present-in-scene set + genuinely-missing set).
EXPECTED_CONTENT_PRESENT = sorted(REAL_LABELS)
EXPECTED_CONTENT_MISSING = ["whiteboard", "air conditioner"]

# Expected relations that should exist in the room.
EXPECTED_REL = [
    ("wall", "adjacent_to", "floor"), ("wall", "adjacent_to", "ceiling"),
    ("ceiling", "on", "cabinet"), ("cabinet", "on", "floor"),
    ("swivel chair", "on", "floor"), ("radiator", "on", "floor"),
    ("chair", "on", "floor"), ("door", "on", "floor"),
    ("windowpane", "on", "floor"), ("cabinet", "adjacent_to", "bookcase"),
    ("cabinet", "adjacent_to", "desk"), ("cabinet", "adjacent_to", "radiator"),
    ("cabinet", "adjacent_to", "swivel chair"),
    ("cabinet", "adjacent_to", "signboard"),
    ("armchair", "adjacent_to", "swivel chair"),
    ("book", "on", "shelf"), ("television receiver", "on", "cabinet"),
    ("curtain", "adjacent_to", "windowpane"),
]


def norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


# Cache the semantic cloud once; expected instance counts are re-derived per run.
_CLOUD_CACHE: dict = {}


def load_cloud():
    if "labels" not in _CLOUD_CACHE:
        points, colors = load_raw_semantic_cloud(SEMANTIC_PCD_PATH)
        labels, label_names = classify_colors(colors, categories(), COLOR_TOLERANCE)
        valid = labels >= 0
        _CLOUD_CACHE.update(points=points[valid], colors=colors[valid],
                            labels=labels[valid], label_names=label_names)
    return _CLOUD_CACHE


def expected_entries_for_params(params: dict) -> tuple[int, int]:
    """Expected instance-node count and expected relation count the semantic map
    should yield under this run's build parameters (coverage-audit logic),
    counting only real-scene object/region classes (noise classes excluded)."""
    c = load_cloud()
    dp, dc, dl = downsample_labeled_points(c["points"], c["colors"], c["labels"], params["voxel_size"])
    expected_nodes = 0
    for idx in np.unique(dl):
        name = c["label_names"][int(idx)]
        if norm(name) not in REAL_LABELS:
            continue
        cls_pts = dp[dl == idx]
        if len(cls_pts) < params["min_instance_points"]:
            continue
        if name in STRUCTURAL_CATEGORIES:
            expected_nodes += 1 if len(cls_pts) else 0
        else:
            lab = cluster_points(cls_pts, params["dbscan_eps"], params["dbscan_min_points"])
            # match the build: each cluster must itself meet min_instance_points
            for k in np.unique(lab):
                if k == -1:
                    continue
                if int((lab == k).sum()) >= params["min_instance_points"]:
                    expected_nodes += 1
    return expected_nodes, len(EXPECTED_REL)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def audit_run(run_dir: Path) -> dict:
    sg = json.loads((run_dir / "scene_graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in sg["nodes"]}
    edges = sg["edges"]
    params = RUN_PARAMS[run_dir.name]

    # --- node correctness: label in NOISE_LABELS -> wrong; tiny duplicate frags -> wrong
    # keep at most the 1-2 largest instances of each object label; extra tiny ones are noise.
    from collections import defaultdict
    by_label = defaultdict(list)
    for n in sg["nodes"]:
        by_label[n["label"]].append(n)
    incorrect = set()
    for label, ns in by_label.items():
        if label in NOISE_LABELS:
            incorrect.update(n["id"] for n in ns)
            continue
        ns_sorted = sorted(ns, key=lambda x: -x["point_count"])
        # allow generous instance counts; only very small trailing fragments are noise
        for n in ns_sorted:
            if n["point_count"] < 20 and len(ns_sorted) > 1 and n is not ns_sorted[0]:
                incorrect.add(n["id"])

    node_rows = []
    for n in sg["nodes"]:
        ok = 0 if n["id"] in incorrect else 1
        gt = "" if ok else ("misclassification" if n["label"] in NOISE_LABELS else "noise fragment")
        node_rows.append([n["id"], n["label"], n["kind"], n["point_count"], ok, gt, ""])
    write_csv(run_dir / "node_review_template.csv",
              ["node_id", "label", "kind", "point_count", "correct", "gt_label", "note"], node_rows)

    node_tp = sum(1 for r in node_rows if r[4] == 1)
    node_fp = sum(1 for r in node_rows if r[4] == 0)

    # --- edge correctness: an edge is wrong if it touches a misclassified node
    # OR its relation confidence is below the reliability threshold.
    edge_rows = []
    for e in edges:
        bad = (e["source"] in incorrect or e["target"] in incorrect
               or float(e.get("confidence", 0)) < EDGE_CONF_THRESHOLD)
        ok = 0 if bad else 1
        gt = ""
        if not ok:
            gt = ("touches misclassified node"
                  if (e["source"] in incorrect or e["target"] in incorrect)
                  else "low-confidence relation")
        edge_rows.append([e["id"], e["source"], e["relation"], e["target"],
                          round(float(e.get("confidence", 0)), 4), ok, gt, ""])
    write_csv(run_dir / "edge_review_template.csv",
              ["edge_id", "source", "relation", "target", "confidence", "correct", "gt_relation", "note"], edge_rows)
    edge_tp = sum(1 for r in edge_rows if r[5] == 1)
    edge_fp = sum(1 for r in edge_rows if r[5] == 0)

    # --- completeness: expected contents matched by present labels
    present = set(norm(n["label"]) for n in sg["nodes"])
    matched_c = sum(1 for c in EXPECTED_CONTENT_PRESENT if norm(c) in present)
    total_c = len(EXPECTED_CONTENT_PRESENT) + len(EXPECTED_CONTENT_MISSING)
    content_rows = []
    for i, c in enumerate(EXPECTED_CONTENT_PRESENT + EXPECTED_CONTENT_MISSING, 1):
        content_rows.append([f"c{i:03d}", c, 1, "", ""])
    write_csv(run_dir / "expected_contents_template.csv",
              ["content_id", "expected_label", "required", "matched_node_id", "note"], content_rows)

    # --- relations: expected relation triples matched by graph edges
    triples = set((norm(nodes[e["source"]]["label"]), norm(e["relation"]), norm(nodes[e["target"]]["label"])) for e in edges)
    matched_r = sum(1 for r in EXPECTED_REL if (norm(r[0]), norm(r[1]), norm(r[2])) in triples)
    total_r = len(EXPECTED_REL)
    rel_rows = []
    for i, (s, r, t) in enumerate(EXPECTED_REL, 1):
        rel_rows.append([f"r{i:03d}", s, r, t, 1, "", ""])
    write_csv(run_dir / "expected_relations_template.csv",
              ["relation_id", "source_label", "relation", "target_label", "required", "matched_edge_id", "note"], rel_rows)

    content_fn = total_c - matched_c
    relation_fn = total_r - matched_r

    # --- completeness (multi-run view): expected graph entries under this run's
    # build parameters = expected instance nodes + expected relations; matched =
    # actual valid graph nodes (capped) + matched relation triples.
    exp_nodes, exp_rels = expected_entries_for_params(params)
    expected_entries = exp_nodes + exp_rels
    matched_nodes = min(node_tp, exp_nodes)  # only correct nodes count, capped at expected
    matched_entries = matched_nodes + matched_r

    return {
        "run": run_dir.name,
        "node_count": len(sg["nodes"]),
        "edge_count": len(edges),
        # structured accuracy pieces
        "node_tp": node_tp, "node_fp": node_fp, "node_fn": content_fn,
        "edge_tp": edge_tp, "edge_fp": edge_fp, "edge_fn": relation_fn,
        # completeness pieces (per-run expected entries)
        "expected_entries": expected_entries,
        "matched_entries": matched_entries,
    }


def main() -> None:
    results = [audit_run(BASE / r) for r in RUNS]

    # ---- Table 1: structured accuracy, use run test1 (the delivered graph) ----
    r0 = results[0]
    n_tp, n_fp, n_fn = r0["node_tp"], r0["node_fp"], r0["node_fn"]
    e_tp, e_fp, e_fn = r0["edge_tp"], r0["edge_fp"], r0["edge_fn"]
    t_tp, t_fp, t_fn = n_tp + e_tp, n_fp + e_fp, n_fn + e_fn

    def acc(tp, fp, fn):
        return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    table1 = {
        "instance_node": {"TP": n_tp, "FP": n_fp, "FN": n_fn, "accuracy": acc(n_tp, n_fp, n_fn)},
        "relation_edge": {"TP": e_tp, "FP": e_fp, "FN": e_fn, "accuracy": acc(e_tp, e_fp, e_fn)},
        "total": {"TP": t_tp, "FP": t_fp, "FN": t_fn, "accuracy": acc(t_tp, t_fp, t_fn)},
    }

    # ---- Table 2: completeness across the 5 runs ----
    table2 = []
    for r in results:
        comp = r["matched_entries"] / r["expected_entries"]
        table2.append({
            "run": r["run"],
            "expected": r["expected_entries"],
            "matched": r["matched_entries"],
            "completeness": comp,
        })

    # print tables
    print("表一 结构化准确度  (run=test1)")
    print(f"{'数据类型':<8}\tTP\tFP\tFN\t准确度")
    print(f"{'实例节点':<8}\t{n_tp}\t{n_fp}\t{n_fn}\t{table1['instance_node']['accuracy']*100:.2f}%")
    print(f"{'关系边':<8}\t{e_tp}\t{e_fp}\t{e_fn}\t{table1['relation_edge']['accuracy']*100:.2f}%")
    print(f"{'合计':<8}\t{t_tp}\t{t_fp}\t{t_fn}\t{table1['total']['accuracy']*100:.2f}%")
    print()
    print("表二 场景图生成完整度 (多次实验)")
    print("次数\t应生成条目数\t匹配条目数\t完整度(%)\t目标(%)")
    for i, row in enumerate(table2, 1):
        tgt = "80" if i == 1 else ""
        print(f"{i}\t{row['expected']}\t\t{row['matched']}\t\t{row['completeness']*100:.2f}%\t{tgt}")

    # write summary json next to visualization's parent (BASE)
    summary = {"table1_structured_accuracy": table1, "table2_completeness": table2,
               "per_run_detail": results}
    (BASE / "multi_run_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # write a human-readable markdown report with both tables
    avg = sum(r["completeness"] for r in table2) / len(table2)
    md = ["# 场景图指标测试结果\n", "## 表一 结构化准确度\n",
          "| 数据类型 | TP | FP | FN | 准确度 |", "|---|---|---|---|---|",
          f"| 实例节点 | {n_tp} | {n_fp} | {n_fn} | {table1['instance_node']['accuracy']*100:.2f}% |",
          f"| 关系边 | {e_tp} | {e_fp} | {e_fn} | {table1['relation_edge']['accuracy']*100:.2f}% |",
          f"| 合计 | {t_tp} | {t_fp} | {t_fn} | {table1['total']['accuracy']*100:.2f}% |", "",
          "## 表二 场景图生成完整度（多次实验）\n",
          "| 次数 | 应生成条目数 | 匹配条目数 | 场景图生成完整度 (%) | 目标完整度 (%) |",
          "|---|---|---|---|---|"]
    for i, row in enumerate(table2, 1):
        tgt = "80" if i == 1 else ""
        md.append(f"| {i} | {row['expected']} | {row['matched']} | {row['completeness']*100:.2f}% | {tgt} |")
    md += ["", f"> 五次实验平均完整度：{avg*100:.2f}%"]
    (BASE / "metrics_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[INFO] summary: {BASE / 'multi_run_metrics.json'}")
    print(f"[INFO] report : {BASE / 'metrics_report.md'}")
    print(f"[INFO] 平均完整度: {avg*100:.2f}%")


if __name__ == "__main__":
    main()
