"""
Anomaly Report Generator
=========================
Generates structured reports in JSON and CSV formats containing
all detection results with geographic coordinates, confidence scores,
severity classifications, and summary statistics.
"""

import csv
import json
import io
import base64
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import UPLOADS_DIR


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_thumbnails: bool = True
    thumbnail_size: int = 128
    include_scores_breakdown: bool = True
    csv_delimiter: str = ","


class ReportGenerator:
    """
    Generates structured anomaly reports in JSON and CSV formats.
    
    Reports include:
    - Unique report ID and timestamp
    - Source file information
    - All detection details with coordinates
    - Summary statistics by severity and class
    - Optional base64-encoded thumbnail crops
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    def generate_json_report(
        self,
        geotagged_detections: List,
        source_filename: str,
        image: Optional[np.ndarray] = None,
        processing_time_ms: float = 0.0,
        detection_mode: str = "heuristic",
    ) -> Dict:
        """
        Generate a comprehensive JSON report.

        Args:
            geotagged_detections: List of GeotaggedDetection objects.
            source_filename: Name of the original sonar file.
            image: Original image for thumbnail extraction.
            processing_time_ms: Total processing time.
            detection_mode: 'heuristic' or 'model' — tagged in the output so
                            consumers know whether detections came from a trained
                            model or the classical-CV fallback.

        Returns:
            Dictionary representing the full report.
        """
        now = datetime.now(timezone.utc)
        report_id = f"RPT-{now.strftime('%Y-%m-%d-%H%M%S')}"

        anomalies = []
        for det in geotagged_detections:
            anomaly = det.to_dict()

            # Add thumbnail if image is provided
            if self.config.include_thumbnails and image is not None:
                thumbnail = self._extract_thumbnail(image, det.bbox_px)
                if thumbnail is not None:
                    anomaly["thumbnail"] = thumbnail

            anomalies.append(anomaly)

        # Compute summary statistics
        summary = self._compute_summary(geotagged_detections)

        report = {
            "report_id": report_id,
            "generated_at": now.isoformat(),
            "source_file": source_filename,
            "processing_time_ms": round(processing_time_ms, 2),
            "detection_mode": detection_mode,
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "summary": summary,
        }

        # Warn consumers when results are from the heuristic (non-model) detector
        if "heuristic" in detection_mode:
            report["heuristic_warning"] = (
                "Results produced by physics-aware heuristic detector — NOT a "
                "trained YOLOv8 model. Accuracy is lower and class labels are "
                "approximate. See TRAINING_GUIDE.md to train a real model."
            )

        return report

    def generate_csv_report(
        self,
        geotagged_detections: List,
        source_filename: str,
    ) -> str:
        """
        Generate a flattened CSV report suitable for GIS import.
        
        Returns:
            CSV content as a string.
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.config.csv_delimiter)

        # Header row
        headers = [
            "detection_id",
            "class_name",
            "class_label",
            "confidence",
            "severity",
            "latitude",
            "longitude",
            "depth_m",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_width",
            "bbox_height",
            "est_length_m",
            "est_width_m",
            "source_file",
        ]

        if self.config.include_scores_breakdown:
            headers.extend([
                "yolo_confidence",
                "shadow_score",
                "morphology_score",
                "texture_score",
            ])

        writer.writerow(headers)

        # Data rows
        for det in geotagged_detections:
            row = [
                det.detection_id,
                det.class_name,
                det.class_label,
                round(det.confidence, 1),
                det.severity,
                round(det.location.latitude, 7) if det.location else "",
                round(det.location.longitude, 7) if det.location else "",
                round(det.location.depth_m, 1) if det.location else "",
                det.bbox_px.get("x1", ""),
                det.bbox_px.get("y1", ""),
                det.bbox_px.get("x2", ""),
                det.bbox_px.get("y2", ""),
                det.bbox_px.get("width", ""),
                det.bbox_px.get("height", ""),
                det.estimated_size_m.get("length_m", ""),
                det.estimated_size_m.get("width_m", ""),
                source_filename,
            ]

            if self.config.include_scores_breakdown:
                # ScoredDetection stores sub-scores as individual fields
                row.extend([
                    round(getattr(det, "yolo_confidence", 0) * 100, 1),
                    round(getattr(det, "shadow_score", 0) * 100, 1),
                    round(getattr(det, "morphology_score", 0) * 100, 1),
                    round(getattr(det, "texture_score", 0) * 100, 1),
                ])

            writer.writerow(row)

        return output.getvalue()

    def generate_geojson_report(
        self,
        geotagged_detections: List,
        source_filename: str,
    ) -> Dict:
        """
        Generate a GeoJSON FeatureCollection report suitable for GIS (QGIS, ArcGIS).
        
        Args:
            geotagged_detections: List of GeotaggedDetection objects.
            source_filename: Name of the original sonar file.

        Returns:
            Dict representing GeoJSON FeatureCollection.
        """
        now = datetime.now(timezone.utc)
        features = []

        for det in geotagged_detections:
            lat = det.location.latitude if det.location else None
            lon = det.location.longitude if det.location else None
            if lat is not None and lon is not None:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {
                        "detection_id": det.detection_id,
                        "class_name": det.class_name,
                        "class_label": det.class_label,
                        "confidence": round(det.confidence, 1),
                        "severity": det.severity,
                        "depth_m": det.location.depth_m if det.location else 0.0,
                        "est_length_m": det.estimated_size_m.get("length_m", 0.0),
                        "est_width_m": det.estimated_size_m.get("width_m", 0.0),
                        "source_file": source_filename,
                        "scores": det.scores,
                    },
                })

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "source_file": source_filename,
                "total_features": len(features),
                "generated_at": now.isoformat(),
            },
        }

    def save_report(
        self,
        report: Dict,
        job_id: str,
        format: str = "json",
    ) -> str:
        """
        Save report to disk and return the file path.
        
        Args:
            report: The report dictionary (for JSON/GeoJSON) or CSV string.
            job_id: Unique job identifier.
            format: 'json', 'csv', or 'geojson'.
            
        Returns:
            Path to the saved report file.
        """
        results_dir = UPLOADS_DIR / "results"
        results_dir.mkdir(exist_ok=True)

        if format == "json":
            filepath = results_dir / f"{job_id}_report.json"
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "geojson":
            filepath = results_dir / f"{job_id}_report.geojson"
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            filepath = results_dir / f"{job_id}_report.csv"
            with open(filepath, "w", newline="") as f:
                f.write(report if isinstance(report, str) else json.dumps(report))
        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(filepath)

    def _extract_thumbnail(
        self, image: np.ndarray, bbox: Dict
    ) -> Optional[str]:
        """
        Extract and encode a thumbnail crop of a detection.
        Returns base64-encoded JPEG string.
        """
        try:
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)

            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = image[y1:y2, x1:x2]

            # Resize to thumbnail size
            size = self.config.thumbnail_size
            crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

            # Encode to JPEG base64
            _, buffer = cv2.imencode(".jpg", crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode("utf-8")

            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    def _compute_summary(self, detections: List) -> Dict:
        """Compute summary statistics for the report."""
        if not detections:
            return {
                "by_severity": {"critical": 0, "moderate": 0, "low": 0},
                "by_class": {},
                "avg_confidence": 0,
            }

        by_severity = {"critical": 0, "moderate": 0, "low": 0, "negligible": 0}
        by_class = {}
        confidences = []

        for det in detections:
            sev = det.severity
            if sev in by_severity:
                by_severity[sev] += 1
            by_class[det.class_name] = by_class.get(det.class_name, 0) + 1
            confidences.append(det.confidence)

        # Remove zero-count severities
        by_severity = {k: v for k, v in by_severity.items() if v > 0}

        return {
            "by_severity": by_severity,
            "by_class": by_class,
            "avg_confidence": round(float(np.mean(confidences)), 1) if confidences else 0,
            "max_confidence": round(float(np.max(confidences)), 1) if confidences else 0,
        }
