"""Evaluate checkpoints and create qualitative comparison grids."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import UNIFIED_CLASSES, build_dataloaders, resolve_project_path
from models.model import UNet
from scripts.train import load_config, make_criterion, validate
from utils.metrics import logits_to_predictions
from utils.visualization import save_prediction_grid


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model = UNet(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_checkpoint(cfg: dict, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    _, val_loader, metadata = build_dataloaders(cfg, project_root=PROJECT_ROOT)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    eval_cfg = {**cfg, "model": {**cfg["model"], **model_cfg, "loss": cfg["model"].get("loss", "cross_entropy")}}
    model = load_checkpoint_model(checkpoint_path, device)
    criterion = make_criterion(eval_cfg, device)
    metrics = validate(model, val_loader, criterion, device, int(metadata["num_classes"]))
    return {
        "checkpoint": checkpoint_path.as_posix(),
        "miou": float(metrics["miou"]),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "loss": float(metrics["loss"]),
    }


@torch.no_grad()
def save_qualitative_comparison(
    cfg: dict,
    checkpoint_paths: List[Path],
    output_path: Path,
    device: torch.device,
    max_items: int,
) -> None:
    _, val_loader, _ = build_dataloaders(cfg, project_root=PROJECT_ROOT)
    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    predictions: Dict[str, torch.Tensor] = {}
    for checkpoint_path in checkpoint_paths:
        model = load_checkpoint_model(checkpoint_path, device)
        logits = model(images)
        name = checkpoint_path.stem.replace("_best", "")
        predictions[name] = logits_to_predictions(logits).cpu()
    save_prediction_grid(
        batch,
        predictions,
        output_path,
        mean=cfg["data"].get("mean", [0.485, 0.456, 0.406]),
        std=cfg["data"].get("std", [0.229, 0.224, 0.225]),
        max_items=max_items,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Relative checkpoint paths.")
    parser.add_argument("--max-items", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(resolve_project_path(args.config, PROJECT_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [resolve_project_path(p, PROJECT_ROOT) for p in args.checkpoints]
    out_dir = resolve_project_path(cfg["paths"].get("results_dir", "results"), PROJECT_ROOT) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_checkpoint(cfg, p, device) for p in checkpoints]
    with open(out_dir / "checkpoint_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "miou", "pixel_accuracy", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    save_qualitative_comparison(cfg, checkpoints, out_dir / "qualitative_comparison.png", device, args.max_items)
    print(f"Saved evaluation files to {out_dir.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
