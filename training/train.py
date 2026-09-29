"""
YOLOv8 Training Pipeline for Side-Scan Sonar
==============================================
Prepares datasets and fine-tunes YOLOv8 on sonar imagery.

Usage:
    python training/train.py --data data/sonar_dataset --epochs 100 --device cuda

Dataset Structure (YOLO format):
    data/sonar_dataset/
        images/
            train/  *.png / *.tiff
            val/    *.png / *.tiff
            test/   *.png / *.tiff
        labels/
            train/  *.txt  (YOLO format: class cx cy w h, normalized)
            val/    *.txt
            test/   *.txt
        dataset.yaml
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, CLASS_LABELS, MODELS_DIR, BASE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# ─── Dataset YAML Generator ───────────────────────────────────────────────────

def create_dataset_yaml(dataset_dir: Path) -> str:
    """Generate a YOLOv8-compatible dataset.yaml file."""
    yaml_content = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
    }
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    logger.info(f"Dataset YAML written: {yaml_path}")
    return str(yaml_path)


# ─── Synthetic Dataset Generator (for testing without real data) ───────────────

def generate_synthetic_dataset(
    output_dir: Path,
    n_train: int = 200,
    n_val: int = 50,
    n_test: int = 30,
    image_size: int = 640,
):
    """
    Generate a synthetic sonar dataset for pipeline testing.

    Creates realistic-looking sonar imagery with geometric objects
    that approximate how marine debris appears in SSS.

    NOT a substitute for real labeled sonar data — use only for
    development and pipeline validation.
    """
    logger.info(f"Generating synthetic dataset → {output_dir}")

    splits = {"train": n_train, "val": n_val, "test": n_test}
    rng = np.random.default_rng(42)

    for split, count in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img, labels = _generate_sonar_sample(rng, image_size)
            img_path = img_dir / f"sonar_{split}_{idx:04d}.png"
            lbl_path = lbl_dir / f"sonar_{split}_{idx:04d}.txt"
            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                for lbl in labels:
                    f.write(" ".join(f"{v:.6f}" for v in lbl) + "\n")

        logger.info(f"  {split}: {count} images generated")

    create_dataset_yaml(output_dir)
    logger.info("Synthetic dataset generation complete.")


def _generate_sonar_sample(rng, size: int):
    """Generate one synthetic sonar image with random debris objects."""
    # Background: dark seafloor with low-intensity texture
    bg = rng.integers(20, 55, (size, size), dtype=np.uint8)
    # Add speckle noise
    noise = rng.integers(0, 15, (size, size), dtype=np.uint8)
    img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Nadir strip (dark center band)
    nadir_w = int(size * 0.04)
    cx = size // 2
    img[:, cx - nadir_w // 2: cx + nadir_w // 2] = rng.integers(5, 20)

    labels = []
    n_objects = rng.integers(1, 6)

    for _ in range(n_objects):
        cls_id = int(rng.integers(0, len(CLASS_NAMES)))
        placed, lbl = _place_object(rng, img, cls_id, size)
        if placed:
            labels.append(lbl)

    return img, labels


def _place_object(rng, img: np.ndarray, cls_id: int, size: int):
    """Place a synthetic debris object and return its YOLO label."""
    # Object appearance depends on class
    # Shapes: (shape_type, width_range, height_range) — both are (min, max) tuples
    shapes = {
        0: ("blob",       (20, 50),  (20, 50)),   # ghost_net — irregular blob
        1: ("large_blob", (60, 120), (60, 120)),  # shipwreck — large
        2: ("rect",       (40, 80),  (8,  20)),   # pipe — elongated rectangle
        3: ("circle",     (18, 35),  (18, 35)),   # cylinder — compact circle
        4: ("blob",       (30, 70),  (30, 70)),   # debris_cluster — medium blob
        5: ("line",       (80, 140), (4,  10)),   # cable — thin line
        6: ("blob",       (10, 28),  (10, 28)),   # anomaly — small
    }

    shape_type, w_range, h_range = shapes.get(cls_id, ("blob", (20, 40), (20, 40)))

    # Random position (avoid nadir center band)
    cx_range = [size * 0.05, size * 0.45]  # port side
    if rng.random() > 0.5:
        cx_range = [size * 0.55, size * 0.95]  # stbd side

    cx = int(rng.uniform(*cx_range))
    cy = int(rng.uniform(size * 0.05, size * 0.95))

    # Ensure low < high for rng.integers (add +1 guard)
    w_lo, w_hi = w_range
    h_lo, h_hi = h_range
    w = int(rng.integers(w_lo, max(w_lo + 1, w_hi)))
    h = int(rng.integers(h_lo, max(h_lo + 1, h_hi)))

    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(size - 1, cx + w // 2)
    y2 = min(size - 1, cy + h // 2)

    if x2 <= x1 or y2 <= y1:
        return False, None

    # Intensity: objects are brighter than background
    intensity = int(rng.integers(160, 240))

    if shape_type == "circle":
        cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, intensity, -1)
        # Add acoustic shadow below
        shadow_y1 = min(size - 1, y2)
        shadow_y2 = min(size - 1, y2 + h // 2)
        if shadow_y2 > shadow_y1:
            img[shadow_y1:shadow_y2, x1:x2] = np.clip(
                img[shadow_y1:shadow_y2, x1:x2].astype(int) - 40, 5, 255
            )
    elif shape_type == "line":
        angle = rng.uniform(0, 180)
        rad = np.radians(angle)
        dx = int((w / 2) * np.cos(rad))
        dy = int((w / 2) * np.sin(rad))
        pt1 = (int(np.clip(cx - dx, 0, size - 1)), int(np.clip(cy - dy, 0, size - 1)))
        pt2 = (int(np.clip(cx + dx, 0, size - 1)), int(np.clip(cy + dy, 0, size - 1)))
        thickness = max(2, h)
        cv2.line(img, pt1, pt2, intensity, thickness)
        pad = thickness // 2 + 2
        x1 = max(0, min(pt1[0], pt2[0]) - pad)
        y1 = max(0, min(pt1[1], pt2[1]) - pad)
        x2 = min(size - 1, max(pt1[0], pt2[0]) + pad)
        y2 = min(size - 1, max(pt1[1], pt2[1]) + pad)
    elif shape_type in ("rect", "large_blob", "blob"):
        cv2.rectangle(img, (x1, y1), (x2, y2), intensity, -1)
        # Gaussian blur edges for realism
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (5, 5), 0)
        # Shadow
        s_y1 = min(size - 1, y2)
        s_y2 = min(size - 1, y2 + h // 3)
        if s_y2 > s_y1:
            img[s_y1:s_y2, x1:x2] = np.clip(
                img[s_y1:s_y2, x1:x2].astype(int) - 35, 5, 255
            )

    # YOLO label (normalized)
    bw = (x2 - x1) / size
    bh = (y2 - y1) / size
    bcx = ((x1 + x2) / 2) / size
    bcy = ((y1 + y2) / 2) / size
    return True, [cls_id, bcx, bcy, bw, bh]


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    data_yaml: str,
    base_weights: str = "yolov8s.pt",
    epochs: int = 100,
    image_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    output_name: str = "yolov8_sonar",
):
    """
    Fine-tune YOLOv8 on sonar dataset.

    Args:
        data_yaml:    Path to dataset.yaml
        base_weights: Starting weights (yolov8s.pt, yolov8n.pt, or custom)
        epochs:       Number of training epochs
        image_size:   Input resolution (640 recommended, 416 for edge)
        batch_size:   Batch size (reduce if OOM)
        device:       'cpu', 'cuda', '0' (GPU index), or 'auto'
        output_name:  Name for saved weights file
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed: pip install ultralytics")
        return

    logger.info(f"Starting training: {base_weights} → sonar fine-tune")
    logger.info(f"  Data: {data_yaml}")
    logger.info(f"  Epochs: {epochs} | Batch: {batch_size} | Size: {image_size}")
    logger.info(f"  Device: {device}")

    model = YOLO(base_weights)

    # Use absolute path for project so YOLOv8 doesn't nest it under runs/detect/
    train_project_dir = str(BASE_DIR / "training" / "runs")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device if device != "auto" else None,
        project=train_project_dir,
        name=output_name,
        exist_ok=True,
        # Sonar-specific augmentation settings
        augment=True,
        hsv_h=0.0,       # Sonar is grayscale — no hue augmentation
        hsv_s=0.0,
        hsv_v=0.4,        # Vary intensity (simulates different sonar gains)
        flipud=0.3,       # Flip vertically (sonar can be port/stbd mirrored)
        fliplr=0.5,
        mosaic=0.8,
        degrees=5.0,      # Small rotation (vessel pitch)
        scale=0.3,
        translate=0.1,
        # Training stability
        patience=20,      # Early stopping
        save_period=10,
        verbose=True,
    )

    # Resolve best weights from the actual save directory reported by ultralytics
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (
        BASE_DIR / "training" / "runs" / output_name
    )
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        dest = MODELS_DIR / f"{output_name}.pt"
        shutil.copy2(best_weights, dest)
        logger.info(f"Best weights saved to: {dest}")
    else:
        # Fallback: scan for best.pt anywhere under training/runs
        found = list((BASE_DIR / "training" / "runs").rglob("best.pt"))
        if found:
            best_weights = found[-1]
            MODELS_DIR.mkdir(exist_ok=True)
            dest = MODELS_DIR / f"{output_name}.pt"
            shutil.copy2(best_weights, dest)
            logger.info(f"Best weights saved to: {dest} (located at {best_weights})")
        else:
            logger.warning("Best weights not found after training")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on sonar dataset")
    parser.add_argument("--data", default="data/sonar_dataset",
                        help="Dataset directory (will be created with synthetic data if missing)")
    parser.add_argument("--weights", default="yolov8s.pt",
                        help="Base YOLOv8 weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto",
                        help="Training device: cpu, cuda, 0, auto")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic training data (for testing)")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=100)
    args = parser.parse_args()

    dataset_dir = Path(args.data)

    if args.synthetic or not dataset_dir.exists():
        logger.info("Generating synthetic training data...")
        generate_synthetic_dataset(
            dataset_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            image_size=args.imgsz,
        )

    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        logger.error(f"dataset.yaml not found at {yaml_path}")
        sys.exit(1)

    train(
        data_yaml=str(yaml_path),
        base_weights=args.weights,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
    )
