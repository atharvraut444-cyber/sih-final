"""
Post-Training Evaluation Script
=================================
Evaluates a trained YOLOv8 sonar model and produces:
  - mAP50 / mAP50-95 per class and overall
  - Precision-Recall curves (saved as PNG)
  - Confusion matrix heatmap (saved as PNG)
  - Inference speed benchmark (ms/image)
  - Per-class detection summary report

Usage:
    python training/evaluate.py --weights models/yolov8_sonar.pt --data data/sonar_dataset/dataset.yaml
    python training/evaluate.py --weights models/yolov8_sonar.pt --data data/sonar_dataset/dataset.yaml --split val
    python training/evaluate.py --weights models/yolov8_sonar.pt --data data/sonar_dataset/dataset.yaml --bench
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, CLASS_LABELS, CLASS_COLORS, MODELS_DIR, BASE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# --- Helpers -----------------------------------------------------------------

def _color_hex_to_bgr(hex_color: str):
    """Convert '#rrggbb' to (B, G, R) tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


# --- Evaluation --------------------------------------------------------------

def run_validation(weights: str, data_yaml: str, split: str = "val",
                   image_size: int = 640, device: str = "cpu",
                   output_dir: Path = None):
    """
    Run YOLOv8 validation and return metrics dict.

    Args:
        weights:    Path to trained .pt weights
        data_yaml:  Path to dataset.yaml
        split:      Dataset split to evaluate ('val' or 'test')
        image_size: Inference resolution
        device:     'cpu', '0' (GPU), etc.
        output_dir: Where to save plots and reports

    Returns:
        metrics dict with mAP, precision, recall, and per-class data
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed: pip install ultralytics")
        return {}

    if output_dir is None:
        output_dir = BASE_DIR / "training" / "eval_results"
    else:
        # Make absolute relative to project root if not already absolute
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {weights}")
    model = YOLO(weights)

    logger.info(f"Validating on split='{split}' | imgsz={image_size} | device={device}")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=image_size,
        device=device if device != "auto" else None,
        save_json=True,
        plots=True,
        project=str(output_dir),
        name="val_run",
        exist_ok=True,
        verbose=True,
    )

    # Extract metrics
    metrics = {
        "map50":    float(results.box.map50),
        "map5095":  float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
    }

    # Per-class breakdown
    per_class = {}
    names = model.names
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        for idx, cls_id in enumerate(results.box.ap_class_index):
            cls_name = names.get(int(cls_id), f"class_{cls_id}")
            per_class[cls_name] = {
                "ap50": float(results.box.ap50[idx]) if results.box.ap50 is not None else 0.0,
                "ap":   float(results.box.ap[idx])   if results.box.ap   is not None else 0.0,
            }

    metrics["per_class"] = per_class

    _print_metrics_table(metrics)
    _save_metrics_report(metrics, output_dir)

    logger.info(f"Plots and reports saved to: {output_dir}")
    return metrics


# --- Inference Speed Benchmark -----------------------------------------------

def benchmark_speed(weights: str, image_size: int = 640, n_runs: int = 100,
                    device: str = "cpu") -> dict:
    """
    Measure inference latency over n_runs random sonar images.

    Returns dict with mean_ms, std_ms, fps.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return {}

    logger.info(f"Speed benchmark: {n_runs} runs on {device} | imgsz={image_size}")
    model = YOLO(weights)
    rng = np.random.default_rng(42)

    # Warm-up
    for _ in range(5):
        dummy = rng.integers(20, 200, (image_size, image_size), dtype=np.uint8)
        dummy_bgr = cv2.cvtColor(dummy, cv2.COLOR_GRAY2BGR)
        model.predict(dummy_bgr, imgsz=image_size, verbose=False,
                      device=device if device != "auto" else None)

    times = []
    for _ in range(n_runs):
        dummy = rng.integers(20, 200, (image_size, image_size), dtype=np.uint8)
        dummy_bgr = cv2.cvtColor(dummy, cv2.COLOR_GRAY2BGR)
        t0 = time.perf_counter()
        model.predict(dummy_bgr, imgsz=image_size, verbose=False,
                      device=device if device != "auto" else None)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    fps     = 1000.0 / mean_ms

    logger.info(f"Inference latency: {mean_ms:.1f} ms +/- {std_ms:.1f} ms  ({fps:.1f} FPS)")
    return {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}


# --- Reporting ---------------------------------------------------------------

def _print_metrics_table(metrics: dict):
    """Pretty-print metrics summary to logger."""
    logger.info("=" * 55)
    logger.info(f"  Overall mAP@0.5      : {metrics.get('map50',    0) * 100:.1f}%")
    logger.info(f"  Overall mAP@0.5:0.95 : {metrics.get('map5095',  0) * 100:.1f}%")
    logger.info(f"  Mean Precision        : {metrics.get('precision', 0) * 100:.1f}%")
    logger.info(f"  Mean Recall           : {metrics.get('recall',    0) * 100:.1f}%")
    logger.info("-" * 55)
    logger.info(f"  {'Class':<22}  {'AP@50':>8}  {'AP@.5:.95':>10}")
    logger.info(f"  {'-'*22}  {'--------':>8}  {'----------':>10}")
    for cls_name, vals in metrics.get("per_class", {}).items():
        label = CLASS_LABELS.get(_cls_id_from_name(cls_name), cls_name)[:22]
        logger.info(f"  {label:<22}  {vals['ap50']*100:>7.1f}%  {vals['ap']*100:>9.1f}%")
    logger.info("=" * 55)


def _cls_id_from_name(name: str) -> int:
    """Reverse-lookup class ID from short name."""
    for cid, cname in CLASS_NAMES.items():
        if cname == name:
            return cid
    return -1


def _save_metrics_report(metrics: dict, output_dir: Path):
    """Save metrics as a plain-text report file."""
    report_path = output_dir / "evaluation_report.txt"
    lines = [
        "YOLOv8 Sonar Model Evaluation Report",
        "=" * 50,
        f"Overall mAP@0.5      : {metrics.get('map50',    0) * 100:.2f}%",
        f"Overall mAP@0.5:0.95 : {metrics.get('map5095',  0) * 100:.2f}%",
        f"Mean Precision        : {metrics.get('precision', 0) * 100:.2f}%",
        f"Mean Recall           : {metrics.get('recall',    0) * 100:.2f}%",
        "",
        "Per-Class AP:",
        "-" * 50,
    ]
    for cls_name, vals in metrics.get("per_class", {}).items():
        label = CLASS_LABELS.get(_cls_id_from_name(cls_name), cls_name)
        lines.append(f"  {label:<30} AP50={vals['ap50']*100:.1f}%  AP={vals['ap']*100:.1f}%")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Evaluation report saved: {report_path}")


def save_sample_detections(weights: str, image_dir: str, output_dir: Path,
                           image_size: int = 640, n_samples: int = 12,
                           conf: float = 0.25, device: str = "cpu"):
    """
    Run inference on sample images and save annotated results.

    Useful for visual quality inspection after training.

    Args:
        weights:    Trained .pt weights path
        image_dir:  Directory with .jpg/.png images to visualize
        output_dir: Where to save annotated images
        image_size: Inference resolution
        n_samples:  Max images to process
        conf:       Detection confidence threshold
        device:     Inference device
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return

    model = YOLO(weights)
    img_dir = Path(image_dir)
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    imgs = imgs[:n_samples]

    if not imgs:
        logger.warning(f"No images found in {image_dir}")
        return

    vis_dir = output_dir / "sample_detections"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        results = model.predict(img, imgsz=image_size, conf=conf, verbose=False,
                                device=device if device != "auto" else None)
        annotated = results[0].plot()
        out_path = vis_dir / f"det_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

    logger.info(f"Sample detections saved to: {vis_dir}")


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained sonar YOLOv8 model")
    parser.add_argument("--weights", default=str(MODELS_DIR / "yolov8_sonar.pt"),
                        help="Path to trained .pt weights")
    parser.add_argument("--data", required=True,
                        help="Path to dataset.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu",
                        help="'cpu', 'cuda', '0', or 'auto'")
    parser.add_argument("--output", default="training/eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--bench", action="store_true",
                        help="Run inference speed benchmark")
    parser.add_argument("--vis-dir", default=None,
                        help="Optional: directory of images for visual detection samples")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        logger.info("Train first: python training/train.py --synthetic")
        sys.exit(1)

    output_dir = Path(args.output)

    # Run validation metrics
    metrics = run_validation(
        weights=str(weights_path),
        data_yaml=args.data,
        split=args.split,
        image_size=args.imgsz,
        device=args.device,
        output_dir=output_dir,
    )

    # Optional: speed benchmark
    if args.bench:
        speed = benchmark_speed(
            weights=str(weights_path),
            image_size=args.imgsz,
            device=args.device,
        )
        if speed:
            with open(output_dir / "speed_benchmark.txt", "w") as f:
                f.write(f"Mean latency : {speed['mean_ms']:.1f} ms\n")
                f.write(f"Std          : {speed['std_ms']:.1f} ms\n")
                f.write(f"FPS          : {speed['fps']:.1f}\n")

    # Optional: visual sample detections
    if args.vis_dir:
        save_sample_detections(
            weights=str(weights_path),
            image_dir=args.vis_dir,
            output_dir=output_dir,
            image_size=args.imgsz,
            device=args.device,
        )
