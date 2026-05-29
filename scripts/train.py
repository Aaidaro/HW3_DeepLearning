"""Training utilities for the segmentation experiments."""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import UNIFIED_CLASSES, build_dataloaders, count_pixels_by_class, resolve_project_path
from models.model import build_model
from utils.metrics import SegmentationConfusionMatrix, logits_to_predictions
from utils.visualization import plot_class_distribution, plot_loss_curves, plot_miou_curve


class FocalLoss(nn.Module):
    """Manual multi-class focal loss for semantic segmentation.

    Focal loss down-weights easy pixels and focuses training on hard or rare pixels.
    logits shape: [N, C, H, W], targets shape: [N, H, W].
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    requested = str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def case_name(case_cfg: dict) -> str:
    up = case_cfg.get("upsampling", "bilinear")
    skip = "skip" if bool(case_cfg.get("use_skip", True)) else "noskip"
    bn = "bn" if bool(case_cfg.get("batch_norm", False)) else "nobn"
    loss = case_cfg.get("loss", "cross_entropy")
    return f"{up}_{skip}_{bn}_{loss}"


def make_criterion(cfg: dict, device: torch.device) -> nn.Module:
    loss_name = str(cfg["model"].get("loss", "cross_entropy"))
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "focal":
        alpha_values = cfg["training"].get("focal_alpha", None)
        alpha = None
        if alpha_values is not None:
            alpha = torch.tensor(alpha_values, dtype=torch.float32, device=device)
        return FocalLoss(gamma=float(cfg["training"].get("focal_gamma", 2.0)), alpha=alpha)
    raise ValueError(f"Unknown loss: {loss_name}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    return running_loss / max(sample_count, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    cm = SegmentationConfusionMatrix(num_classes)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        preds = logits_to_predictions(logits)
        cm.update(preds, masks)
        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    metrics = cm.compute()
    metrics["loss"] = running_loss / max(sample_count, 1)
    return metrics


def update_summary(summary_path: Path, row: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Dict[str, object]] = {}
    if summary_path.exists():
        with open(summary_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                rows[old["case_name"]] = dict(old)
    rows[str(row["case_name"])] = row
    fieldnames = [
        "case_name",
        "upsampling",
        "use_skip",
        "batch_norm",
        "loss",
        "best_val_miou",
        "best_epoch",
        "checkpoint",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({name: rows[key].get(name, "") for name in fieldnames})


def run_experiment(base_cfg: dict, case_cfg: dict, project_root: Path = PROJECT_ROOT) -> Dict[str, object]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy with simple types
    cfg["model"].update(case_cfg)
    set_seed(int(cfg["training"].get("seed", 42)))
    device = get_device(cfg)

    train_loader, val_loader, metadata = build_dataloaders(cfg, project_root=project_root)
    num_classes = int(metadata["num_classes"])
    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    criterion = make_criterion(cfg, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-3)))

    name = case_name(cfg["model"])
    output_dir = resolve_project_path(cfg["paths"].get("results_dir", "results"), project_root) / "segmentation" / name
    checkpoint_dir = resolve_project_path(cfg["paths"].get("saved_models_dir", "models/saved_models"), project_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "used_config.yaml")

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pixel_accuracy": []}
    best_miou = -1.0
    best_epoch = -1
    best_checkpoint = checkpoint_dir / f"{name}_best.pt"
    epochs = int(cfg["training"].get("epochs", 30))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device, num_classes)
        val_loss = float(val_metrics["loss"])
        val_miou = float(val_metrics["miou"])
        val_acc = float(val_metrics["pixel_accuracy"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_pixel_accuracy"].append(val_acc)

        print(
            f"[{name}] epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.model_config,
                    "classes": UNIFIED_CLASSES,
                    "epoch": epoch,
                    "best_val_miou": best_miou,
                },
                best_checkpoint,
            )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_loss_curves(history, output_dir / "loss_curves.png")
    plot_miou_curve(history, output_dir / "miou_curve.png")

    summary_row = {
        "case_name": name,
        "upsampling": cfg["model"].get("upsampling"),
        "use_skip": cfg["model"].get("use_skip"),
        "batch_norm": cfg["model"].get("batch_norm"),
        "loss": cfg["model"].get("loss"),
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "checkpoint": best_checkpoint.relative_to(project_root).as_posix(),
    }
    update_summary(resolve_project_path(cfg["paths"].get("summary_csv", "results/segmentation_summary.csv"), project_root), summary_row)
    return summary_row


def run_data_analysis(cfg: dict, project_root: Path = PROJECT_ROOT) -> None:
    train_loader, val_loader, metadata = build_dataloaders(cfg, project_root=project_root)
    train_counts = count_pixels_by_class(train_loader.dataset, int(metadata["num_classes"]))
    val_counts = count_pixels_by_class(val_loader.dataset, int(metadata["num_classes"]))
    out_dir = resolve_project_path(cfg["paths"].get("results_dir", "results"), project_root) / "analysis"
    plot_class_distribution(
        train_counts,
        val_counts,
        out_dir / "class_distribution_train_val.png",
        out_dir / "class_distribution_train_val.csv",
    )
    print(f"Saved class distribution plot and CSV to {out_dir.relative_to(project_root).as_posix()}")


def read_summary(summary_path: Path) -> List[Dict[str, str]]:
    if not summary_path.exists():
        return []
    with open(summary_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def best_case_from_summary(cfg: dict, project_root: Path = PROJECT_ROOT) -> Dict[str, object]:
    summary_path = resolve_project_path(cfg["paths"].get("summary_csv", "results/segmentation_summary.csv"), project_root)
    rows = read_summary(summary_path)
    if not rows:
        raise RuntimeError("No previous experiment summary found. Run baseline4 first.")
    best = max(rows, key=lambda r: float(r["best_val_miou"]))
    return {
        "upsampling": best["upsampling"],
        "use_skip": str(best["use_skip"]).lower() == "true",
        "batch_norm": str(best["batch_norm"]).lower() == "true",
        "loss": best["loss"],
    }
