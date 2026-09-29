"""
Detection Pipeline Orchestrator
=================================
Ties together all core modules into a single end-to-end function:

  Raw image bytes
      ↓  SonarPreprocessor     — despeckle, CLAHE, tile
      ↓  SonarDetector         — YOLOv8 tiled inference
      ↓  ConfidenceScorer      — shadow/morphology/texture scoring
      ↓  GeotaggingEngine      — pixel → GPS coordinates
      ↓  ReportGenerator       — JSON + CSV reports + annotated image
      →  PipelineResult
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import UPLOADS_DIR, FREQUENCY_SONAR_RANGE_M
from core.preprocessor import SonarPreprocessor
from core.detector import SonarDetector
from core.confidence import ConfidenceScorer
from core.geotagging import GeotaggingEngine
from core.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """All outputs from a single detection run."""
    job_id: str
    json_report: Dict
    csv_report: str
    annotated_image: np.ndarray
    annotated_image_path: str
    json_report_path: str
    csv_report_path: str
    total_time_ms: float
    clean_image_path: Optional[str] = None
    geojson_report: Optional[Dict] = None
    geojson_report_path: Optional[str] = None


class DetectionPipeline:
    """
    End-to-end sonar debris detection pipeline.

    A single instance is shared across all requests.
    The underlying YOLOv8 model is loaded once at startup.
    """

    def __init__(self):
        logger.info("Initializing detection pipeline...")
        self.preprocessor = SonarPreprocessor()
        self.preprocessor.warm_start()
        self.detector = SonarDetector()
        self.scorer = ConfidenceScorer()
        self.geo_engine = GeotaggingEngine()
        self.reporter = ReportGenerator()
        logger.info(
            f"Pipeline ready — model: {self.detector.model_name} "
            f"| mode: {self.detector.detection_mode}"
        )
        if self.detector.demo_mode:
            logger.warning(
                "=" * 60
            )
            logger.warning(
                "⚠  HEURISTIC MODE ACTIVE — no trained model weights found."
            )
            logger.warning(
                "   All detections use a classical-CV heuristic, NOT YOLOv8."
            )
            logger.warning(
                "   See TRAINING_GUIDE.md to train a real sonar model."
            )
            logger.warning(
                "=" * 60
            )

    @property
    def model_info(self) -> Dict:
        return {
            "model_name": self.detector.model_name,
            "detection_mode": self.detector.detection_mode,
            "demo_mode": self.detector.demo_mode,
            "confidence_threshold": 0.35,
        }

    def run(
        self,
        image_bytes: bytes,
        job_id: str,
        source_filename: str,
        origin_lat: float = 12.9716,
        origin_lon: float = 77.5946,
        heading_deg: float = 0.0,
        sonar_range_m: float = 75.0,
        towfish_altitude_m: float = 0.0,
        layback_m: float = 0.0,
        frequency: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline on raw image bytes.

        Args:
            image_bytes:     Raw bytes of the uploaded sonar image.
            job_id:          Unique identifier for this job.
            source_filename: Original uploaded filename.
            origin_lat:      Survey origin latitude.
            origin_lon:      Survey origin longitude.
            heading_deg:     Vessel heading in degrees.
            sonar_range_m:   One-side sonar swath in metres.
            towfish_altitude_m: Towfish altitude above seabed (for slant-range correction).
            layback_m:       Towfish layback distance behind vessel GPS.
            frequency:       Operating frequency (e.g. '100kHz', '300kHz').

        Returns:
            PipelineResult with all outputs and file paths.
        """
        start_total = time.time()

        # ── Step 1: Decode image from bytes ──────────────────────────
        nparr = np.frombuffer(image_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise ValueError("Could not decode uploaded image. Check the file format.")

        logger.info(f"[{job_id}] Image decoded: {raw_img.shape}")

        # ── Step 2: Preprocess ────────────────────────────────────────
        t0 = time.time()
        prep_result = self.preprocessor.process_array(raw_img)
        logger.info(
            f"[{job_id}] Preprocessed in {(time.time()-t0)*1000:.1f}ms "
            f"— {prep_result.metadata['num_tiles']} tile(s)"
        )

        # ── Step 3: Detect ────────────────────────────────────
        t0 = time.time()
        det_result = self.detector.detect_tiled(
            tiles=prep_result.tiles,
            tile_positions=prep_result.tile_positions,
            full_image_shape=prep_result.processed.shape[:2],
            frequency_hint=frequency,
        )
        logger.info(
            f"[{job_id}] Detected {det_result.count} candidates "
            f"in {(time.time()-t0)*1000:.1f}ms"
        )

        # ── Step 4: Score & filter ────────────────────────────────────
        t0 = time.time()
        scored = self.scorer.score_detections(
            detections=det_result.detections,
            image=prep_result.processed,
            shadow_map=prep_result.shadow_map,
        )
        filtered = self.scorer.filter_detections(scored)
        logger.info(
            f"[{job_id}] After scoring: {len(scored)} scored, "
            f"{len(filtered)} kept — {(time.time()-t0)*1000:.1f}ms"
        )

        # ── Step 5: Geotag (Slant-range & Layback corrected) ──────────
        if frequency and frequency in FREQUENCY_SONAR_RANGE_M:
            self.geo_engine.sonar_range_m = FREQUENCY_SONAR_RANGE_M[frequency]
        else:
            self.geo_engine.sonar_range_m = sonar_range_m

        geotagged = self.geo_engine.geotag_detections(
            scored_detections=filtered,
            image_shape=prep_result.processed.shape[:2],
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            heading_deg=heading_deg,
            towfish_altitude_m=towfish_altitude_m,
            layback_m=layback_m,
            frequency=frequency,
        )
        logger.info(f"[{job_id}] Geotagged {len(geotagged)} detections")

        # ── Step 6: Annotate full image ───────────────────────────────
        annotated = self._draw_on_full_image(prep_result.processed, filtered)

        # ── Step 7: Generate reports ──────────────────────────────────
        total_ms = (time.time() - start_total) * 1000

        json_report = self.reporter.generate_json_report(
            geotagged_detections=geotagged,
            source_filename=source_filename,
            image=prep_result.processed,
            processing_time_ms=total_ms,
            detection_mode=self.detector.detection_mode,
        )
        csv_report = self.reporter.generate_csv_report(
            geotagged_detections=geotagged,
            source_filename=source_filename,
        )
        geojson_report = self.reporter.generate_geojson_report(
            geotagged_detections=geotagged,
            source_filename=source_filename,
        )

        # ── Step 8: Save to disk ──────────────────────────────────────
        results_dir = UPLOADS_DIR / "results"
        results_dir.mkdir(exist_ok=True)

        # Clean preprocessed image (without burned-in annotations)
        clean_path = str(results_dir / f"{job_id}_clean.jpg")
        cv2.imwrite(clean_path, prep_result.processed)

        # Annotated image
        annotated_path = str(results_dir / f"{job_id}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)

        # JSON report
        json_path = self.reporter.save_report(json_report, job_id, "json")

        # CSV report
        csv_path = self.reporter.save_report(csv_report, job_id, "csv")

        # GeoJSON report
        geojson_path = self.reporter.save_report(geojson_report, job_id, "geojson")

        logger.info(
            f"[{job_id}] Pipeline complete in {total_ms:.1f}ms "
            f"— {len(geotagged)} anomalies found"
        )

        return PipelineResult(
            job_id=job_id,
            json_report=json_report,
            csv_report=csv_report,
            annotated_image=annotated,
            annotated_image_path=annotated_path,
            json_report_path=json_path,
            csv_report_path=csv_path,
            total_time_ms=total_ms,
            clean_image_path=clean_path,
            geojson_report=geojson_report,
            geojson_report_path=geojson_path,
        )

    def _draw_on_full_image(
        self, image: np.ndarray, scored_detections: list
    ) -> np.ndarray:
        """
        Draw scored detection boxes on the full processed image.
        Colors boxes by severity (red=critical, orange=moderate, green=low).
        """
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        SEVERITY_COLORS = {
            "critical": (57, 71, 255),    # #ff4739 → BGR
            "moderate": (2, 165, 255),    # #ffa502 → BGR
            "low": (115, 213, 46),        # #2ed573 → BGR
            "negligible": (140, 125, 116), # #747d8c → BGR
        }

        for det in scored_detections:
            x1, y1, x2, y2 = det.bbox
            bgr = SEVERITY_COLORS.get(det.severity, (255, 255, 255))

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

            # Label: "ghost_net 87.3%"
            label = f"{det.class_name}  {det.final_confidence:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Background rect for label
            label_y = max(y1 - lh - baseline - 4, 0)
            cv2.rectangle(
                annotated,
                (x1, label_y),
                (x1 + lw + 4, label_y + lh + baseline + 4),
                bgr,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y + lh + 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated


# Singleton — loaded once at startup, reused for all requests
_pipeline: Optional[DetectionPipeline] = None


def get_pipeline() -> DetectionPipeline:
    """FastAPI dependency — returns the shared pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline()
    return _pipeline
