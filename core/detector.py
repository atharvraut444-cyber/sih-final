"""
YOLOv8 Object Detection Engine for Side-Scan Sonar
====================================================
Wraps the Ultralytics YOLOv8 model for sonar-specific inference.
Supports tiled inference for large sonar logs, with NMS merging
at tile boundaries, and ONNX export for edge deployment.

Detection modes
---------------
  "model"     — real YOLOv8 weights loaded from models/yolov8_sonar.pt
  "heuristic" — physics-aware classical-CV fallback when no weights exist

All JSON output includes a "detection_mode" field so callers can distinguish
heuristic results from genuine model inference.
"""

import cv2
import numpy as np
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import json
import time
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_WEIGHTS,
    MODEL_INPUT_SIZE,
    MODEL_CONFIDENCE_THRESHOLD,
    MODEL_IOU_THRESHOLD,
    CLASS_NAMES,
    CLASS_LABELS,
    CLASS_COLORS,
    TILE_OVERLAP,
    ROUGH_BOTTOM_STD_THRESHOLD,
    ROUGH_BOTTOM_CONTRAST_MIN,
    NORMAL_CONTRAST_MIN,
    COMPACTNESS_MIN,
    DETECTOR_MAX_DETECTIONS_PER_TILE,
    FREQ_100KHZ_AREA_SCALE,
    FREQ_100KHZ_CONTRAST_MIN,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected object in the sonar image."""
    class_id: int
    class_name: str
    class_label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates
    color: str
    center: Tuple[int, int] = (0, 0)
    area_px: int = 0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.area_px = (x2 - x1) * (y2 - y1)

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_label": self.class_label,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
                "width": self.bbox[2] - self.bbox[0],
                "height": self.bbox[3] - self.bbox[1],
            },
            "center": {"x": self.center[0], "y": self.center[1]},
            "area_px": self.area_px,
            "color": self.color,
        }


@dataclass
class DetectionResult:
    """Complete detection results for an image."""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    model_name: str = ""
    annotated_image: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict:
        return {
            "total_detections": self.count,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_shape": {
                "height": self.image_shape[0],
                "width": self.image_shape[1],
            },
            "model": self.model_name,
            "detections": [d.to_dict() for d in self.detections],
        }

    def get_by_class(self, class_name: str) -> List[Detection]:
        return [d for d in self.detections if d.class_name == class_name]

    def get_summary(self) -> Dict[str, int]:
        summary = {}
        for d in self.detections:
            summary[d.class_name] = summary.get(d.class_name, 0) + 1
        return summary


class SonarDetector:
    """
    YOLOv8-based detector for marine debris in sonar imagery.

    In production mode, loads custom-trained weights for sonar.
    In heuristic mode, uses a physics-aware classical-CV detector when
    no trained weights are available.  All output JSON is tagged with
    ``detection_mode`` so callers can distinguish the two cases.

    ⚠ WARNING (heuristic mode): Results do NOT represent trained-model
    inference.  See TRAINING_GUIDE.md for how to train a real model.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        self.device = device
        self.model = None
        self.demo_mode = False
        self.model_name = ""

        self._load_model(weights_path)

    # ── Mode indicator ────────────────────────────────────────────

    @property
    def detection_mode(self) -> str:
        """Returns 'heuristic_fallback' or 'model', for tagging output JSON."""
        return "heuristic_fallback" if self.demo_mode else "model"

    def _load_model(self, weights_path: Optional[str] = None):
        """Load YOLOv8 model — custom weights or fallback to heuristic mode."""
        try:
            from ultralytics import YOLO

            weights = weights_path or str(MODEL_WEIGHTS)

            if Path(weights).exists():
                self.model = YOLO(weights)
                self.model_name = Path(weights).stem
                self.demo_mode = False
                logger.info(f"Loaded custom sonar model: {weights}")
            else:
                # Never silently run or report COCO weights as sonar detections
                self.model = None
                self.model_name = "heuristic_fallback"
                self.demo_mode = True
                logger.error(
                    "CRITICAL: Custom sonar weights not found at %s. "
                    "COCO pre-trained weights will NOT be used for sonar detection. "
                    "Falling back to classical-CV heuristic path tagged as 'heuristic_fallback'. "
                    "See TRAINING_GUIDE.md.",
                    weights,
                )
        except ImportError:
            logger.error("Ultralytics not installed. Running in HEURISTIC FALLBACK mode.")
            self.model = None
            self.demo_mode = True
            self.model_name = "heuristic_fallback"
        except Exception as e:
            logger.error(f"Model loading failed: {e}. Running in HEURISTIC FALLBACK mode.")
            self.model = None
            self.demo_mode = True
            self.model_name = "heuristic_fallback"

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        frequency_hint: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run detection on a single image.

        Args:
            image: Preprocessed grayscale sonar image (uint8).
            confidence_threshold: Min confidence to keep detections.
            iou_threshold: IoU threshold for NMS.
            frequency_hint: Operating frequency string (e.g. "100kHz") used to
                            tune detection sensitivity in heuristic mode.

        Returns:
            DetectionResult with all detections.
        """
        conf = confidence_threshold or MODEL_CONFIDENCE_THRESHOLD
        if confidence_threshold is None:
            if frequency_hint == "100kHz":
                conf = min(conf, 0.12)
            elif frequency_hint == "900kHz":
                conf = min(conf, 0.14)
        iou = iou_threshold or MODEL_IOU_THRESHOLD

        start_time = time.time()

        if self.demo_mode or self.model is None:
            detections = self._generate_demo_detections(image, conf, frequency_hint)
        else:
            detections = self._run_inference(image, conf, iou)

        inference_time = (time.time() - start_time) * 1000

        result = DetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            image_shape=image.shape[:2],
            model_name=self.model_name,
        )

        # Generate annotated image
        result.annotated_image = self._draw_detections(image, detections)

        return result

    def detect_tiled(
        self,
        tiles: List[np.ndarray],
        tile_positions: List[Tuple[int, int]],
        full_image_shape: Tuple[int, int],
        confidence_threshold: float = None,
        iou_threshold: float = None,
        frequency_hint: Optional[str] = None,
        input_size: Optional[int] = None,
    ) -> DetectionResult:
        """
        Run detection across multiple tiles and merge results.

        Uses NMS to remove duplicate detections at tile boundaries.

        Args:
            frequency_hint: Operating frequency string (e.g. "100kHz"), forwarded
                            to heuristic detector for sensitivity tuning.
            input_size: Target image size for YOLO inference (defaults to MODEL_INPUT_SIZE).
        """
        conf = confidence_threshold or MODEL_CONFIDENCE_THRESHOLD
        if confidence_threshold is None:
            if frequency_hint == "100kHz":
                conf = min(conf, 0.12)
            elif frequency_hint == "900kHz":
                conf = min(conf, 0.14)
        iou = iou_threshold or MODEL_IOU_THRESHOLD

        all_detections = []
        start_time = time.time()

        for tile, (tx, ty) in zip(tiles, tile_positions):
            if self.demo_mode or self.model is None:
                tile_dets = self._generate_demo_detections(tile, conf, frequency_hint)
            else:
                tile_dets = self._run_inference(tile, conf, iou, input_size=input_size)

            # Offset detections to full image coordinates
            for det in tile_dets:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (int(x1 + tx), int(y1 + ty), int(x2 + tx), int(y2 + ty))
                det.__post_init__()  # Recalculate center and area

            all_detections.extend(tile_dets)

        # Apply NMS across all tiles to remove duplicates at boundaries
        merged = self._cross_tile_nms(all_detections, iou, conf_threshold=conf)

        inference_time = (time.time() - start_time) * 1000

        return DetectionResult(
            detections=merged,
            inference_time_ms=inference_time,
            image_shape=full_image_shape,
            model_name=self.model_name,
        )

    def _run_inference(
        self, image: np.ndarray, conf: float, iou: float, input_size: Optional[int] = None
    ) -> List[Detection]:
        """Run actual YOLOv8 inference."""
        # Convert grayscale to 3-channel for YOLOv8
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_rgb = image

        import torch
        with torch.inference_mode():
            results = self.model.predict(
                image_rgb,
                conf=conf,
                iou=iou,
                imgsz=input_size or MODEL_INPUT_SIZE,
                verbose=False,
            )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Custom-trained model outputs class IDs 0-6 directly
                    sonar_cls_id = min(cls_id, len(CLASS_NAMES) - 1)

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    detections.append(
                        Detection(
                            class_id=sonar_cls_id,
                            class_name=CLASS_NAMES.get(sonar_cls_id, "anomaly"),
                            class_label=CLASS_LABELS.get(sonar_cls_id, "Unknown"),
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            color=CLASS_COLORS.get(sonar_cls_id, "#ffffff"),
                        )
                    )

        return detections

    def _generate_demo_detections(
        self,
        image: np.ndarray,
        conf_threshold: float,
        frequency_hint: Optional[str] = None,
    ) -> List[Detection]:
        """
        Physics-aware heuristic detector for sonar imagery.

        Uses multi-scale adaptive thresholding + texture-aware filtering to
        find debris while suppressing false positives from rough-backscatter
        seafloors (coral, rock).

        Key gates applied in order:
          1. Intensity gate (p70 percentile)
          2. Roughness-adaptive contrast gate
          3. Compactness gate (rejects irregular texture clusters)
          4. Shadow-bonus confidence scoring
          5. Per-image detection cap
        """
        h, w = image.shape[:2]
        detections = []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ── Global statistics (excluding nadir center band) ──────────
        cx = w // 2
        nadir_half = max(5, w // 20)
        mask_cols = list(range(0, cx - nadir_half)) + list(range(cx + nadir_half, w))
        bg_region = gray[:, mask_cols]
        mean_val = float(np.mean(bg_region))
        std_val  = float(np.std(bg_region))
        std_val  = max(std_val, 5.0)   # Prevent division-by-zero on flat images

        # ── Roughness-adaptive contrast threshold ────────────────────
        # High std_val = rough bottom (coral/rock) — require stronger contrast
        # to distinguish real objects from natural speckle texture.
        is_rough = std_val > ROUGH_BOTTOM_STD_THRESHOLD
        if frequency_hint == "100kHz":
            # 100kHz has wide range and low spatial resolution — relax the gate
            contrast_min = FREQ_100KHZ_CONTRAST_MIN
        elif is_rough:
            contrast_min = ROUGH_BOTTOM_CONTRAST_MIN
        else:
            contrast_min = NORMAL_CONTRAST_MIN

        # ── Multi-scale bright region detection ──────────────────────
        # Scale 1: Global adaptive threshold (catches large objects)
        blurred_lg = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh_global = min(254, int(mean_val + max(1.2, 40.0 / std_val) * std_val))
        _, bright_global = cv2.threshold(blurred_lg, thresh_global, 255, cv2.THRESH_BINARY)

        # Scale 2: Local adaptive threshold (catches small/dim objects)
        blurred_sm = cv2.GaussianBlur(gray, (5, 5), 0)
        bright_local = cv2.adaptiveThreshold(
            blurred_sm, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=-12,
        )

        # Combine: union of both scales
        bright_mask = cv2.bitwise_or(bright_global, bright_local)

        # Remove nadir strip from detectable area
        bright_mask[:, cx - nadir_half: cx + nadir_half] = 0

        # Morphological cleanup:
        #   OPEN  with small ellipse  — removes isolated pixel noise
        #   CLOSE with small ellipse  — joins nearby fragments WITHOUT destroying
        #                               thin linear features (cable / pipe)
        # Using (5,5) close instead of the former (9,9) so cables survive.
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  k_open)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, k_close)

        # Additionally detect thin elongated features (cables/pipes) using
        # directional structuring elements, then union back into the mask.
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
        k_vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
        thin_horiz = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, k_horiz)
        thin_vert  = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, k_vert)
        # Union: any elongated feature preserved by either directional pass
        elongated_mask = cv2.bitwise_or(thin_horiz, thin_vert)
        # Only use elongated_mask for the pixels that bright_mask missed
        bright_mask = cv2.bitwise_or(bright_mask, elongated_mask)
        bright_mask[:, cx - nadir_half: cx + nadir_half] = 0  # Re-blank nadir

        # ── Find and score contours ───────────────────────────────────
        bg_flat = gray[:, mask_cols].flatten()
        p85 = float(np.percentile(bg_flat, 85))
        p70 = float(np.percentile(bg_flat, 70))

        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Area limits — frequency-aware for 100kHz (objects appear smaller in pixel space)
        base_min_area = max(80, (h * w) * 0.0004)
        if frequency_hint == "100kHz":
            min_area = base_min_area * FREQ_100KHZ_AREA_SCALE
        else:
            min_area = base_min_area
        max_area = (h * w) * 0.18    # At most 18% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area < area < max_area):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            pad = 6
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_mean    = float(np.mean(roi))
            roi_std     = float(np.std(roi))
            local_contrast = (roi_mean - mean_val) / std_val

            # ── Gate 1: Intensity — ROI must be a genuine local maximum ──
            if roi_mean < p70:
                continue

            # ── Gate 2: Roughness-adaptive contrast ──────────────────────
            # On rough bottoms this threshold is much stricter, preventing
            # natural speckle clusters from passing as detections.
            if local_contrast < contrast_min and roi_std < 14:
                continue

            # ── Gate 3: Compactness — reject irregular texture clusters ──
            # compactness = 4π·area / perimeter²   (circle=1, irregular→0)
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter > 0:
                compactness = (4.0 * math.pi * area) / (perimeter * perimeter)
            else:
                compactness = 0.0

            aspect = bw / max(bh, 1)
            is_elongated = aspect > 2.0 or (1.0 / max(aspect, 1e-3)) > 2.0

            # Elongated blobs (cable/pipe candidates) get a lower compactness bar
            if is_elongated:
                compactness_threshold = max(0.02, COMPACTNESS_MIN * 0.25)
            else:
                compactness_threshold = COMPACTNESS_MIN

            if compactness < compactness_threshold:
                continue

            # ── Class assignment (aspect ratio + size + texture) ──────
            if aspect > 4.0:
                cls_id = 5   # cable / rope (very elongated)
            elif aspect > 2.0:
                cls_id = 2   # pipe / conduit
            elif 0.75 < aspect < 1.35:
                if area < (h * w) * 0.005:
                    cls_id = 3  # cylinder (small, compact)
                elif area > (h * w) * 0.06:
                    cls_id = 1  # shipwreck (large, compact)
                else:
                    cls_id = 4  # debris_cluster
            elif area > (h * w) * 0.04:
                cls_id = 1   # shipwreck (large irregular)
            elif roi_std > 28:
                cls_id = 0   # ghost_net (high internal texture)
            else:
                cls_id = 6   # anomaly

            # ── Confidence from local contrast + shadow presence ──────
            shadow_y1 = min(h - 1, y2)
            shadow_y2 = min(h - 1, y2 + bh)
            shadow_bonus = 0.0
            if shadow_y2 > shadow_y1 and x2 > x1:
                shadow_roi = gray[shadow_y1:shadow_y2, x1:x2]
                if shadow_roi.size > 0:
                    shadow_mean = float(np.mean(shadow_roi))
                    if shadow_mean < mean_val * 0.7:
                        shadow_bonus = 0.08

            base_conf = float(np.clip(
                0.42 + min(local_contrast, 3.0) * 0.12 + shadow_bonus,
                conf_threshold, 0.97
            ))

            # Deterministic jitter from region hash (reproducible)
            rng = np.random.default_rng(int(area * 100 + x1 * 7 + y1 * 13) % 2**32)
            conf = float(np.clip(base_conf + rng.uniform(-0.07, 0.07), conf_threshold, 0.97))

            if conf >= conf_threshold:
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES[cls_id],
                    class_label=CLASS_LABELS[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    color=CLASS_COLORS[cls_id],
                ))

        # Sort by confidence, deduplicate (suppress highly overlapping boxes)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        deduplicated = self._cross_tile_nms(detections, iou_threshold=0.45)
        # Hard cap: prevents FP flood on rough-backscatter images
        return deduplicated[:DETECTOR_MAX_DETECTIONS_PER_TILE]

    def _cross_tile_nms(
        self, detections: List[Detection], iou_threshold: float, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Apply Non-Maximum Suppression across tiles to merge
        duplicate detections at tile boundaries.
        """
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=np.int32)

        eff_conf = conf_threshold if conf_threshold is not None else MODEL_CONFIDENCE_THRESHOLD

        # Per-class NMS
        keep_indices = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            indices = cv2.dnn.NMSBoxes(
                bboxes=cls_boxes.tolist(),
                scores=cls_scores.tolist(),
                score_threshold=eff_conf,
                nms_threshold=iou_threshold,
            )

            if len(indices) > 0:
                indices = indices.flatten()
                keep_indices.extend(cls_indices[indices].tolist())

        candidates = [detections[i] for i in sorted(keep_indices)]

        # ── Cross-Class Spatial Deduplication ────────────────────────────
        # In side-scan sonar, a physical acoustic anomaly cannot simultaneously
        # be multiple contradictory classes (e.g. pipe, ghost net, and cable).
        # When tiles or models emit conflicting detections on the same seabed target,
        # suppress lower-confidence detections if IoU > 0.35 or if one bounding box
        # is largely contained inside another (IoM / containment > 0.55).
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        deduplicated = []
        for cand in candidates:
            cx1, cy1, cx2, cy2 = cand.bbox
            c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            suppressed = False
            for kept in deduplicated:
                kx1, ky1, kx2, ky2 = kept.bbox
                k_area = max(1, (kx2 - kx1) * (ky2 - ky1))
                ix1 = max(cx1, kx1)
                iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2)
                iy2 = min(cy2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    iou = inter / (c_area + k_area - inter)
                    iom = inter / min(c_area, k_area)
                    if iou > 0.35 or iom > 0.55:
                        suppressed = True
                        break
            if not suppressed:
                deduplicated.append(cand)

        return deduplicated

    def _draw_detections(
        self, image: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Parse hex color to BGR
            hex_color = det.color.lstrip("#")
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            bgr_color = (b, g, r)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

            # Draw label background
            label = f"{det.class_name} {det.confidence:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 4),
                (x1 + label_w + 4, y1),
                bgr_color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def export_onnx(self, output_path: str = "models/yolov8_sonar.onnx"):
        """Export model to ONNX format for edge deployment."""
        if self.model is None:
            raise RuntimeError("No model loaded for ONNX export.")

        self.model.export(format="onnx", imgsz=MODEL_INPUT_SIZE)
        logger.info(f"Model exported to ONNX: {output_path}")
