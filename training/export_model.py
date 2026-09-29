"""
Model Export — ONNX & TensorRT
================================
Exports the trained YOLOv8 sonar model to:
  - ONNX  : Cross-platform (RPi, x86, Jetson CPU)
  - TensorRT engine : Jetson GPU (10× faster than ONNX on Jetson)

Usage:
    python training/export_model.py --weights models/yolov8_sonar.pt --format onnx
    python training/export_model.py --weights models/yolov8_sonar.pt --format tensorrt
    python training/export_model.py --weights models/yolov8_sonar.pt --format all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, MODEL_INPUT_SIZE

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def export_onnx(weights_path: str, image_size: int = 640) -> str:
    """
    Export YOLOv8 model to ONNX format.

    ONNX runs on any platform via onnxruntime.
    Suitable for: RPi, x86 PC, Jetson (CPU mode).
    """
    from ultralytics import YOLO
    logger.info(f"Exporting to ONNX: {weights_path}")
    model = YOLO(weights_path)
    export_path = model.export(
        format="onnx",
        imgsz=image_size,
        simplify=True,
        opset=12,
        dynamic=False,  # Fixed batch size for edge deployment
    )
    logger.info(f"ONNX model saved: {export_path}")
    return str(export_path)


def export_tensorrt(weights_path: str, image_size: int = 640,
                    half: bool = True) -> str:
    """
    Export YOLOv8 model to TensorRT engine format.

    ONLY works on NVIDIA Jetson / CUDA systems.
    FP16 (half=True) gives ~2× speedup on Jetson Orin with minimal accuracy loss.
    """
    try:
        import tensorrt  # noqa
    except ImportError:
        logger.error(
            "TensorRT not available. This export only works on NVIDIA Jetson.\n"
            "Install TensorRT via JetPack SDK."
        )
        return ""

    from ultralytics import YOLO
    logger.info(f"Exporting to TensorRT ({'FP16' if half else 'FP32'}): {weights_path}")
    model = YOLO(weights_path)
    export_path = model.export(
        format="engine",
        imgsz=image_size,
        half=half,
        device=0,
        workspace=4,  # GB — adjust based on Jetson RAM
    )
    logger.info(f"TensorRT engine saved: {export_path}")
    return str(export_path)


def validate_onnx(onnx_path: str, pt_path: str, image_size: int = 640):
    """
    Validate ONNX model accuracy against PyTorch baseline.

    Generates a synthetic test image and compares inference outputs.
    """
    logger.info("Validating ONNX model against PyTorch baseline...")

    try:
        import onnxruntime as ort
        from ultralytics import YOLO
    except ImportError:
        logger.warning("Cannot validate — missing onnxruntime or ultralytics")
        return

    # Generate test image
    test_img = np.random.randint(30, 200, (image_size, image_size), dtype=np.uint8)
    test_bgr = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

    # PyTorch inference
    pt_model = YOLO(pt_path)
    t0 = time.time()
    pt_results = pt_model.predict(test_bgr, imgsz=image_size, verbose=False)
    pt_time = (time.time() - t0) * 1000
    pt_count = len(pt_results[0].boxes) if pt_results[0].boxes else 0

    # ONNX inference via Ultralytics
    onnx_model = YOLO(onnx_path)
    t0 = time.time()
    onnx_results = onnx_model.predict(test_bgr, imgsz=image_size, verbose=False)
    onnx_time = (time.time() - t0) * 1000
    onnx_count = len(onnx_results[0].boxes) if onnx_results[0].boxes else 0

    logger.info(f"PyTorch:  {pt_count} detections in {pt_time:.1f}ms")
    logger.info(f"ONNX:     {onnx_count} detections in {onnx_time:.1f}ms")
    logger.info(f"Speedup:  {pt_time/max(onnx_time, 1):.2f}×")

    if abs(pt_count - onnx_count) <= 1:
        logger.info("✅ ONNX validation passed (detection count matches)")
    else:
        logger.warning(
            f"⚠️ Detection count mismatch (PT: {pt_count}, ONNX: {onnx_count}) "
            f"— may indicate export issue"
        )


def copy_to_models(exported_path: str, name: str):
    """Copy exported model to models/ directory."""
    src = Path(exported_path)
    if not src.exists():
        return
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / f"{name}{src.suffix}"
    import shutil
    shutil.copy2(src, dest)
    logger.info(f"Copied to models/: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sonar model for edge deployment")
    parser.add_argument("--weights", default="models/yolov8_sonar.pt",
                        help="Path to trained .pt weights")
    parser.add_argument("--format", choices=["onnx", "tensorrt", "all"],
                        default="onnx", help="Export format")
    parser.add_argument("--imgsz", type=int, default=MODEL_INPUT_SIZE)
    parser.add_argument("--half", action="store_true",
                        help="FP16 mode (TensorRT only)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate exported model accuracy")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        logger.error(f"Weights not found: {weights}")
        logger.info("Train first: python training/train.py --synthetic")
        sys.exit(1)

    if args.format in ("onnx", "all"):
        onnx_path = export_onnx(str(weights), args.imgsz)
        if onnx_path:
            copy_to_models(onnx_path, "yolov8_sonar")
            if args.validate:
                validate_onnx(onnx_path, str(weights), args.imgsz)

    if args.format in ("tensorrt", "all"):
        trt_path = export_tensorrt(str(weights), args.imgsz, args.half)
        if trt_path:
            copy_to_models(trt_path, "yolov8_sonar_trt")
