"""Metrics for semantic segmentation."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class SegmentationConfusionMatrix:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update with preds/targets shaped [N,H,W]."""
        preds_np = preds.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        mask = (targets_np >= 0) & (targets_np < self.num_classes)
        encoded = self.num_classes * targets_np[mask].astype(np.int64) + preds_np[mask].astype(np.int64)
        hist = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.matrix += hist.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        tp = np.diag(self.matrix).astype(np.float64)
        gt = self.matrix.sum(axis=1).astype(np.float64)
        pred = self.matrix.sum(axis=0).astype(np.float64)
        union = gt + pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        miou = float(np.nanmean(iou))
        pixel_acc = float(tp.sum() / max(self.matrix.sum(), 1))
        return {
            "iou_per_class": iou,
            "miou": miou,
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": self.matrix.copy(),
        }


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def mean_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    cm = SegmentationConfusionMatrix(num_classes)
    cm.update(logits_to_predictions(logits), targets)
    return float(cm.compute()["miou"])
