"""Data discovery, mask remapping, splitting, and dataloaders for the football segmentation task.

All dataset paths are read from config/config.yaml as paths relative to the project root.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

UNIFIED_CLASSES: List[str] = [
    "Background",
    "Player",
    "Goalkeeper",
    "Referee",
    "Ball",
    "Goal Bar",
    "Advertisement",
    "Audience",
    "Staff",
]
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}

# Colors are only used for visualization of unified masks.
UNIFIED_COLORS: Dict[int, Tuple[int, int, int]] = {
    CLASS_TO_INDEX["Background"]: (0, 0, 0),
    CLASS_TO_INDEX["Player"]: (38, 198, 129),
    CLASS_TO_INDEX["Goalkeeper"]: (255, 106, 77),
    CLASS_TO_INDEX["Referee"]: (22, 100, 252),
    CLASS_TO_INDEX["Ball"]: (201, 19, 223),
    CLASS_TO_INDEX["Goal Bar"]: (255, 0, 29),
    CLASS_TO_INDEX["Advertisement"]: (237, 34, 236),
    CLASS_TO_INDEX["Audience"]: (143, 182, 45),
    CLASS_TO_INDEX["Staff"]: (111, 48, 253),
}

# Dataset-1 label colors -> unified classes.
DATASET1_COLOR_TO_CLASS: Dict[Tuple[int, int, int], str] = {
    (237, 34, 236): "Advertisement",
    (201, 158, 74): "Background",  # Field is merged into Background for this exercise.
    (96, 32, 192): "Ball",
    (89, 134, 179): "Goal Bar",
    (153, 223, 219): "Goalkeeper",
    (255, 106, 77): "Goalkeeper",
    (22, 100, 252): "Referee",
    (143, 182, 45): "Audience",
    (38, 198, 129): "Player",
    (27, 154, 218): "Player",
    (0, 0, 0): "Background",
}

# Dataset-2 label colors -> unified classes. Black is Staff in this dataset.
DATASET2_COLOR_TO_CLASS: Dict[Tuple[int, int, int], str] = {
    (137, 126, 126): "Background",  # Ground is merged into Background.
    (255, 160, 1): "Player",
    (254, 233, 3): "Player",
    (255, 159, 0): "Goalkeeper",
    (255, 235, 0): "Goalkeeper",
    (238, 171, 171): "Referee",
    (201, 19, 223): "Ball",
    (255, 0, 29): "Goal Bar",
    (27, 71, 151): "Advertisement",
    (111, 48, 253): "Audience",
    (0, 0, 0): "Staff",
}

DATASET_COLOR_MAPS: Dict[str, Dict[Tuple[int, int, int], int]] = {
    "dataset1": {rgb: CLASS_TO_INDEX[name] for rgb, name in DATASET1_COLOR_TO_CLASS.items()},
    "dataset2": {rgb: CLASS_TO_INDEX[name] for rgb, name in DATASET2_COLOR_TO_CLASS.items()},
}


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: Path
    mask_path: Path
    dataset_name: str


def project_root_from_file() -> Path:
    """Return project root without hard-coding any absolute path."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_like: str | Path, project_root: Optional[Path] = None) -> Path:
    """Resolve a config path relative to the project root and reject hard-coded absolute paths."""
    path = Path(path_like)
    if path.is_absolute():
        raise ValueError(f"Use relative paths in config, got absolute path: {path}")
    root = project_root if project_root is not None else project_root_from_file()
    return root / path


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def discover_dataset1(dataset_root: Path) -> List[SegmentationRecord]:
    """Discover dataset-1 pairs: root/images/name.PNG and root/masks/name.PNG."""
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Dataset 1 should contain images/ and masks/ under {dataset_root.as_posix()}"
        )

    records: List[SegmentationRecord] = []
    for image_path in sorted(image_dir.iterdir(), key=lambda p: p.name.lower()):
        if not _is_image_file(image_path):
            continue
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            # Allow different mask suffix with same stem.
            candidates = sorted(mask_dir.glob(f"{image_path.stem}.*"))
            candidates = [p for p in candidates if _is_image_file(p)]
            if not candidates:
                continue
            mask_path = candidates[0]
        records.append(SegmentationRecord(image_path, mask_path, "dataset1"))
    return records


def _find_image_for_fuse_mask(image_dir: Path, mask_path: Path) -> Optional[Path]:
    """Find the original image corresponding to a Dataset-2 ___fuse mask."""
    base_name = re.split(r"___fuse", mask_path.name, flags=re.IGNORECASE)[0]

    # Exact and case-insensitive filename match, e.g. mask 'Frame.jpg___fuse.PNG'
    # with image 'Frame.JPG'.
    exact = image_dir / base_name
    if exact.exists() and _is_image_file(exact):
        return exact
    for candidate in image_dir.iterdir():
        if candidate.name.lower() == base_name.lower() and _is_image_file(candidate):
            return candidate

    # Fallback: match by stem only.
    wanted_stem = Path(base_name).stem.lower()
    for candidate in image_dir.iterdir():
        if "___" in candidate.name:
            continue
        if candidate.stem.lower() == wanted_stem and _is_image_file(candidate):
            return candidate
    return None


def discover_dataset2(dataset_root: Path) -> List[SegmentationRecord]:
    """Discover dataset-2 pairs from the images/ folder and use only ___fuse masks."""
    image_dir = dataset_root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Dataset 2 should contain images/ under {dataset_root.as_posix()}")

    fuse_masks = [
        p for p in sorted(image_dir.iterdir(), key=lambda p: p.name.lower())
        if _is_image_file(p) and "___fuse" in p.name.lower()
    ]
    records: List[SegmentationRecord] = []
    for mask_path in fuse_masks:
        image_path = _find_image_for_fuse_mask(image_dir, mask_path)
        if image_path is not None:
            records.append(SegmentationRecord(image_path, mask_path, "dataset2"))
    return records


def discover_all_records(cfg: dict, project_root: Optional[Path] = None) -> List[SegmentationRecord]:
    root = project_root if project_root is not None else project_root_from_file()
    dataset_cfg = cfg["data"]
    records: List[SegmentationRecord] = []
    records.extend(discover_dataset1(resolve_project_path(dataset_cfg["dataset1_root"], root)))
    records.extend(discover_dataset2(resolve_project_path(dataset_cfg["dataset2_root"], root)))
    if not records:
        raise RuntimeError("No image-mask pairs were found. Check the relative dataset paths in config.yaml.")
    return records


def split_records_by_dataset(
    records: Sequence[SegmentationRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SegmentationRecord], List[SegmentationRecord]]:
    """Split per dataset so both datasets are represented in validation when possible."""
    rng = random.Random(seed)
    train: List[SegmentationRecord] = []
    val: List[SegmentationRecord] = []
    dataset_names = sorted({r.dataset_name for r in records})
    for dataset_name in dataset_names:
        subset = [r for r in records if r.dataset_name == dataset_name]
        rng.shuffle(subset)
        n_val = max(1, int(round(len(subset) * val_ratio))) if len(subset) > 0 else 0
        n_val = min(n_val, len(subset))
        val.extend(subset[:n_val])
        train.extend(subset[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def rgb_mask_to_class_mask(
    mask: Image.Image,
    dataset_name: str,
    color_tolerance: int = 0,
) -> np.ndarray:
    """Convert an RGB or RGBA color mask to a HxW class-index mask.

    Dataset-2 fuse masks may be RGBA. The fourth channel is an alpha/transparency
    channel and is dropped because semantic segmentation here only needs class color.
    """
    if dataset_name not in DATASET_COLOR_MAPS:
        raise KeyError(f"Unknown dataset_name={dataset_name}")

    rgb = np.asarray(mask.convert("RGBA"), dtype=np.uint8)[..., :3]
    h, w, _ = rgb.shape
    output = np.zeros((h, w), dtype=np.int64)
    known = np.zeros((h, w), dtype=bool)
    mapping = DATASET_COLOR_MAPS[dataset_name]

    for color, class_idx in mapping.items():
        matches = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        output[matches] = class_idx
        known |= matches

    if np.any(~known):
        if color_tolerance <= 0:
            unknown_count = int((~known).sum())
            raise ValueError(
                f"Found {unknown_count} pixels with unknown colors in {dataset_name} mask. "
                "Set data.color_tolerance > 0 only if your masks contain compression/anti-aliasing artifacts."
            )
        colors = np.asarray(list(mapping.keys()), dtype=np.int16)
        class_indices = np.asarray(list(mapping.values()), dtype=np.int64)
        flat = rgb.reshape(-1, 3).astype(np.int32)
        distances = ((flat[:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
        nearest = distances.argmin(axis=1)
        nearest_dist = np.sqrt(distances.min(axis=1))
        assigned = class_indices[nearest].reshape(h, w)
        assignable = nearest_dist.reshape(h, w) <= color_tolerance
        output[~known & assignable] = assigned[~known & assignable]
        still_unknown = ~known & ~assignable
        if np.any(still_unknown):
            raise ValueError(
                f"Found {int(still_unknown.sum())} unknown pixels farther than tolerance={color_tolerance}."
            )

    return output


class FootballSegmentationDataset(Dataset):
    """PyTorch Dataset returning image tensor [3,H,W] and mask tensor [H,W]."""

    def __init__(
        self,
        records: Sequence[SegmentationRecord],
        image_size: int | Tuple[int, int] = 256,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        color_tolerance: int = 0,
    ) -> None:
        self.records = list(records)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.color_tolerance = int(color_tolerance)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        mask_image = Image.open(record.mask_path)

        class_mask = rgb_mask_to_class_mask(
            mask_image,
            dataset_name=record.dataset_name,
            color_tolerance=self.color_tolerance,
        )

        image = image.resize(self.image_size, resample=Image.Resampling.BILINEAR)
        mask = Image.fromarray(class_mask.astype(np.uint8), mode="L")
        mask = mask.resize(self.image_size, resample=Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": record.dataset_name,
            "image_path": record.image_path.as_posix(),
            "mask_path": record.mask_path.as_posix(),
        }


def build_dataloaders(cfg: dict, project_root: Optional[Path] = None) -> Tuple[DataLoader, DataLoader, dict]:
    """Create train/validation dataloaders and return split metadata."""
    root = project_root if project_root is not None else project_root_from_file()
    records = discover_all_records(cfg, root)
    train_records, val_records = split_records_by_dataset(
        records,
        val_ratio=float(cfg["data"].get("val_ratio", 0.1)),
        seed=int(cfg["training"].get("seed", 42)),
    )

    common_dataset_kwargs = dict(
        image_size=tuple(cfg["data"].get("image_size", [256, 256])),
        mean=cfg["data"].get("mean", [0.485, 0.456, 0.406]),
        std=cfg["data"].get("std", [0.229, 0.224, 0.225]),
        color_tolerance=int(cfg["data"].get("color_tolerance", 0)),
    )
    train_dataset = FootballSegmentationDataset(train_records, **common_dataset_kwargs)
    val_dataset = FootballSegmentationDataset(val_records, **common_dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=bool(cfg["training"].get("pin_memory", True)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=bool(cfg["training"].get("pin_memory", True)),
    )

    metadata = {
        "num_classes": len(UNIFIED_CLASSES),
        "classes": UNIFIED_CLASSES,
        "train_records": train_records,
        "val_records": val_records,
    }
    return train_loader, val_loader, metadata


def count_pixels_by_class(dataset: Dataset, num_classes: int) -> np.ndarray:
    """Count class pixels for a dataset after preprocessing and resizing."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        item = dataset[i]
        mask = item["mask"].numpy().reshape(-1)
        counts += np.bincount(mask, minlength=num_classes)[:num_classes]
    return counts
