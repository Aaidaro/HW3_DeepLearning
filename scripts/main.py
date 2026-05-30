"""Main entry point for the segmentation project.

Run from the project root, e.g.:
    python -m scripts.main --mode all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import resolve_project_path
from scripts.train import best_case_from_summary, load_config, run_data_analysis, run_experiment
from scripts.yolo_train import run_yolo_coco8


BASELINE_CASES = [
    {"upsampling": "bilinear", "use_skip": True, "batch_norm": False, "loss": "cross_entropy"},
    {"upsampling": "bilinear", "use_skip": False, "batch_norm": False, "loss": "cross_entropy"},
    {"upsampling": "transpose", "use_skip": True, "batch_norm": False, "loss": "cross_entropy"},
    {"upsampling": "transpose", "use_skip": False, "batch_norm": False, "loss": "cross_entropy"},
]


# def run_baseline4(cfg: dict) -> list[dict]:
#     results = []
#     for case in BASELINE_CASES:
#         results.append(run_experiment(cfg, case, PROJECT_ROOT))
#     return results

def run_baseline4(cfg: dict, case) -> list[dict]:
    return run_experiment(cfg, case, PROJECT_ROOT)

def run_batch_norm_stage(cfg: dict) -> dict:
    best = best_case_from_summary(cfg, PROJECT_ROOT)
    best["batch_norm"] = True
    best["loss"] = "cross_entropy"
    return run_experiment(cfg, best, PROJECT_ROOT)


def run_focal_stage(cfg: dict) -> dict:
    best = best_case_from_summary(cfg, PROJECT_ROOT)
    best["loss"] = "focal"
    return run_experiment(cfg, best, PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode",
        choices=["analysis", "baseline4", "bn", "focal", "all", "yolo", "case1", "case2", "case3", "case4"],
        default="analysis",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config, PROJECT_ROOT))

    if args.mode == "analysis":
        run_data_analysis(cfg, PROJECT_ROOT)

    elif args.mode == "case1":
        run_experiment(cfg, BASELINE_CASES[0],PROJECT_ROOT)
    elif args.mode == "case2":
        run_experiment(cfg, BASELINE_CASES[1], PROJECT_ROOT)
    elif args.mode == "case3":
        run_experiment(cfg, BASELINE_CASES[2], PROJECT_ROOT)
    elif args.mode == "case4":
        run_experiment(cfg, BASELINE_CASES[3], PROJECT_ROOT)

    elif args.mode == "baseline4":
        run_baseline4(cfg)
    elif args.mode == "bn":
        run_batch_norm_stage(cfg)
    elif args.mode == "focal":
        run_focal_stage(cfg)
    elif args.mode == "all":
        run_data_analysis(cfg, PROJECT_ROOT)
        run_baseline4(cfg)
        run_batch_norm_stage(cfg)
        run_focal_stage(cfg)
    elif args.mode == "yolo":
        run_yolo_coco8(cfg, PROJECT_ROOT)


if __name__ == "__main__":
    main()
