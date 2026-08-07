#!/usr/bin/env python3
"""Default scene graph task configuration."""

from pathlib import Path


REPO_ROOT = Path("/data1/user/Dense-Object-level-Mapping")

# SEMANTIC_PCD_PATH = Path(
#     "/data1/user/data/fastlivo_output_qs2_03.17/lidar/res/all_raw_qs2_03.17_color_normal.pcd"
# )
SEMANTIC_PCD_PATH = Path(
    "/data1/user/data/2026.05.28_655/点云图片文件/fastlivo_output_2026.05.29_2slow/lidar/res/all_raw_points_color_normal_no_vote.pcd"
)


# SCENE_GRAPH_OUT_DIR = Path(
#     "/data1/user/data/fastlivo_output_qs2_03.17/lidar/scene_graph/all_raw_qs2_03.17_color_normal_strict"
# )
_SG_BASE = Path(
    "/data1/user/data/2026.05.28_655/点云图片文件/fastlivo_output_2026.05.29_2slow/lidar/scene_graph/all_raw_points_color_normal_strict"
)
# 中期 / 终期 输出到不同目录, 由 BENCHMARK_STAGE 环境变量控制 (见文件末尾)
import os as _os_sg
_STAGE_SG = _os_sg.environ.get("BENCHMARK_STAGE", "final").lower()
if _STAGE_SG == "midterm":
    SCENE_GRAPH_OUT_DIR = _SG_BASE.parent / (_SG_BASE.name + "_midterm")
else:
    SCENE_GRAPH_OUT_DIR = _SG_BASE

VISUALIZATION_TITLE = "飞场语义场景图"

BUILD_PARAMS = {
    "voxel_size": 0.18,
    "min_category_points": 1,
    "min_instance_points": 15,
    "dbscan_eps": 0.25,
    "dbscan_min_points": 5,
    "max_instances_per_class": 5000,
    "relation_confidence": 0.70,
    "min_relation_node_points": 20,
}

# ============ 中期 / 终期 切换 ============
# export BENCHMARK_STAGE=midterm  → TARGET_COMPLETENESS=0.80, TARGET_STRUCTURED_ACCURACY=0.65
# export BENCHMARK_STAGE=final    → TARGET_COMPLETENESS=0.90, TARGET_STRUCTURED_ACCURACY=0.85
import os as _os
_STAGE = _os.environ.get("BENCHMARK_STAGE", "final").lower()
if _STAGE == "midterm":
    TARGET_COMPLETENESS = 0.80
    TARGET_STRUCTURED_ACCURACY = 0.65
else:
    TARGET_COMPLETENESS = 0.90
    TARGET_STRUCTURED_ACCURACY = 0.85
