"""YOLOv8n fine-tuning on coco8 with the first 10 layers frozen."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import resolve_project_path


def _find_coco8_val_image(project_root: Path) -> Optional[Path]:
    """Find a validation image without hard-coded absolute paths."""
    candidates = [
        project_root / "datasets" / "coco8" / "images" / "val",
        project_root.parent / "datasets" / "coco8" / "images" / "val",
        project_root / "coco8" / "images" / "val",
    ]
    for directory in candidates:
        if directory.exists():
            images = sorted(list(directory.glob("*.jpg")) + list(directory.glob("*.png")))
            if images:
                return images[0]

    # Fallback: search near the project, still with paths derived relative to this file.
    for base in [project_root, project_root.parent]:
        for image_path in base.rglob("coco8/images/val/*"):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return image_path
    return None


def run_yolo_coco8(cfg: dict, project_root: Path = PROJECT_ROOT) -> None:
    """Train YOLOv8n on coco8 for 50 epochs, freeze first 10 layers, and save plots/prediction."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install ultralytics") from exc

    yolo_cfg = cfg.get("yolo", {})
    results_root = resolve_project_path(cfg["paths"].get("results_dir", "results"), project_root) / "yolo"
    results_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(yolo_cfg.get("model", "yolov8n.pt")))
    train_results = model.train(
        data=str(yolo_cfg.get("data", "coco8.yaml")),
        epochs=int(yolo_cfg.get("epochs", 50)),
        freeze=int(yolo_cfg.get("freeze", 10)),
        imgsz=int(yolo_cfg.get("imgsz", 640)),
        batch=int(yolo_cfg.get("batch", 16)),
        project=results_root.as_posix(),
        name=str(yolo_cfg.get("name", "coco8_yolov8n_freeze10")),
        exist_ok=True,
        plots=True,
    )

    save_dir = Path(train_results.save_dir)
    results_png = save_dir / "results.png"
    if results_png.exists():
        copied = results_root / "coco8_yolov8n_freeze10_results.png"
        shutil.copy2(results_png, copied)
        print(f"Training curves/results plot: {copied.relative_to(project_root).as_posix()}")
    else:
        print(f"Ultralytics did not produce {results_png.as_posix()} yet. Check {save_dir.as_posix()}.")

    val_image = _find_coco8_val_image(project_root)
    if val_image is None:
        print("Could not locate a coco8 validation image automatically. Pass a source manually to model.predict().")
        return

    best_weights = save_dir / "weights" / "best.pt"
    predictor = YOLO(best_weights.as_posix() if best_weights.exists() else str(yolo_cfg.get("model", "yolov8n.pt")))
    predictor.predict(
        source=val_image.as_posix(),
        project=results_root.as_posix(),
        name="validation_prediction",
        exist_ok=True,
        save=True,
    )
    print(f"Validation prediction saved under {results_root.relative_to(project_root).as_posix()}/validation_prediction")


if __name__ == "__main__":
    import yaml

    with open(resolve_project_path("config/config.yaml", PROJECT_ROOT), "r") as f:
        config = yaml.safe_load(f)
    run_yolo_coco8(config, PROJECT_ROOT)
