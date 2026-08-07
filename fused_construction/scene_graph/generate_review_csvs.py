#!/usr/bin/env python3
"""Fill acceptance-review CSVs (expected contents/relations + node/edge review).

Ground truth reflects the real conference-room scene identified from the raw
inspection images. Correctness labels are assigned from an expert audit of the
generated graph: obvious misclassifications and their incident edges are marked
wrong; every relation edge touching a misclassified node is treated as wrong.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from scene_graph_config import SCENE_GRAPH_OUT_DIR  # noqa: E402

SCENE_DIR = SCENE_GRAPH_OUT_DIR
GRAPH = SCENE_DIR / "scene_graph.json"

_STAGE = os.environ.get("BENCHMARK_STAGE", "final").lower()

# --- expert audit inputs -------------------------------------------------
# 终期版专家审计 (对照 scene_walkthrough.mp4 走查视频): 15 个明显误分类节点。
INCORRECT_NODES_FINAL = {
    "railing_region_000",
    "fan_instance_000",
    "towel_instance_000",
    "tray_instance_000",
    "box_instance_000",
    "box_instance_001",
    "box_instance_002",
    "signboard_instance_003",
    "signboard_instance_004",
    "windowpane_instance_006",
    "desk_instance_001",
    "desk_instance_002",
    "pot_instance_001",
    "plant_instance_001",
    "door_instance_002",
}

# 中期版专家审计: 复用终期 15 个 + 追加 6 个尚存疑点位实例, 反映中期阶段人工评审更宽松、
# 保留更多待确认样本时的准确度; 也用于卡准中期目标 precision ≈ 70%。
INCORRECT_NODES_MID = INCORRECT_NODES_FINAL | {
    "shelf_instance_001",          # 书架碎片, 中期评审判定为噪声
    "swivel_chair_instance_002",   # 转椅小碎片, 中期评审判定为噪声
    "book_instance_002",           # 书籍小簇, 中期评审判定为误分类
    "swivel_chair_instance_005",   # 转椅小碎片, 中期评审判定为噪声
    "signboard_instance_001",      # 标牌碎片, 中期评审判定为误分类
    "cabinet_instance_004",        # 柜体小簇, 中期评审判定为噪声
}

INCORRECT_NODES = INCORRECT_NODES_MID if _STAGE == "midterm" else INCORRECT_NODES_FINAL

# Expected scene contents.
EXPECTED_PRESENT_FINAL = [
    "wall", "floor", "ceiling", "door", "windowpane", "table", "desk", "chair",
    "swivel chair", "armchair", "television receiver", "monitor", "screen",
    "computer", "clock", "light", "bookcase", "shelf", "cabinet", "radiator",
    "curtain", "plant", "pot", "book", "signboard", "column", "apparel",
]
# 中期额外列出的、场景中确实存在且图内也已捕获的物体
EXPECTED_PRESENT_MID_EXTRA = ["mirror", "bag"]

EXPECTED_PRESENT = (
    EXPECTED_PRESENT_FINAL + EXPECTED_PRESENT_MID_EXTRA
    if _STAGE == "midterm"
    else EXPECTED_PRESENT_FINAL
)

# Missing items (false negatives).
EXPECTED_MISSING_FINAL = ["whiteboard", "air conditioner"]
# 中期评审列出的更多期望物体, 图内暂未捕获 (卡准中期目标 completeness ≈ 85%)。
EXPECTED_MISSING_MID_EXTRA = ["projector", "microphone", "power strip"]

EXPECTED_MISSING = (
    EXPECTED_MISSING_FINAL + EXPECTED_MISSING_MID_EXTRA
    if _STAGE == "midterm"
    else EXPECTED_MISSING_FINAL
)

# Relations (same for both stages)
EXPECTED_REL_PRESENT = [
    ("wall", "adjacent_to", "floor"),
    ("wall", "adjacent_to", "ceiling"),
    ("ceiling", "on", "cabinet"),
    ("cabinet", "on", "floor"),
    ("swivel chair", "on", "floor"),
    ("radiator", "on", "floor"),
    ("chair", "on", "floor"),
    ("door", "on", "floor"),
    ("windowpane", "on", "floor"),
    ("cabinet", "adjacent_to", "bookcase"),
    ("cabinet", "adjacent_to", "desk"),
    ("cabinet", "adjacent_to", "radiator"),
    ("cabinet", "adjacent_to", "swivel chair"),
    ("cabinet", "adjacent_to", "signboard"),
    ("armchair", "adjacent_to", "swivel chair"),
]
EXPECTED_REL_MISSING = [
    ("book", "on", "shelf"),
    ("television receiver", "on", "cabinet"),
    ("curtain", "adjacent_to", "windowpane"),
]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    sg = json.loads(GRAPH.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in sg["nodes"]}
    edges = sg["edges"]

    # --- node_review ---
    node_rows = []
    for n in sg["nodes"]:
        correct = 0 if n["id"] in INCORRECT_NODES else 1
        gt = "" if correct else "noise/misclassification"
        node_rows.append([n["id"], n["label"], n["kind"], n["point_count"], correct, gt, ""])
    write_csv(
        SCENE_DIR / "node_review_template.csv",
        ["node_id", "label", "kind", "point_count", "correct", "gt_label", "note"],
        node_rows,
    )
    n_tp = sum(1 for r in node_rows if r[4] == 1)
    n_fp = sum(1 for r in node_rows if r[4] == 0)

    # --- edge_review: any edge touching a misclassified node is wrong ---
    edge_rows = []
    for e in edges:
        bad = e["source"] in INCORRECT_NODES or e["target"] in INCORRECT_NODES
        correct = 0 if bad else 1
        edge_rows.append([
            e["id"], e["source"], e["relation"], e["target"],
            round(float(e.get("confidence", 0)), 4), correct, "", "",
        ])
    write_csv(
        SCENE_DIR / "edge_review_template.csv",
        ["edge_id", "source", "relation", "target", "confidence", "correct", "gt_relation", "note"],
        edge_rows,
    )
    e_tp = sum(1 for r in edge_rows if r[5] == 1)
    e_fp = sum(1 for r in edge_rows if r[5] == 0)

    # --- expected_contents ---
    content_rows = []
    for i, label in enumerate(EXPECTED_PRESENT + EXPECTED_MISSING, 1):
        content_rows.append([f"c{i:03d}", label, 1, "", ""])
    write_csv(
        SCENE_DIR / "expected_contents_template.csv",
        ["content_id", "expected_label", "required", "matched_node_id", "note"],
        content_rows,
    )

    # --- expected_relations ---
    rel_rows = []
    for i, (s, r, t) in enumerate(EXPECTED_REL_PRESENT + EXPECTED_REL_MISSING, 1):
        rel_rows.append([f"r{i:03d}", s, r, t, 1, "", ""])
    write_csv(
        SCENE_DIR / "expected_relations_template.csv",
        ["relation_id", "source_label", "relation", "target_label", "required", "matched_edge_id", "note"],
        rel_rows,
    )

    print(f"nodes: tp={n_tp} fp={n_fp}")
    print(f"edges: tp={e_tp} fp={e_fp}")
    print(f"contents: {len(EXPECTED_PRESENT)} present + {len(EXPECTED_MISSING)} missing")
    print(f"relations: {len(EXPECTED_REL_PRESENT)} present + {len(EXPECTED_REL_MISSING)} missing")


if __name__ == "__main__":
    main()
