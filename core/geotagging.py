"""
Geotagging Engine
==================
Maps pixel coordinates in sonar images to real-world GPS locations.
Supports multiple metadata formats: CSV coordinate files, manual entry,
and interpolated track lines.
"""

import csv
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from config import (
    DEFAULT_SONAR_RANGE_M,
    DEFAULT_PING_INTERVAL_M,
    DEFAULT_SPEED_KNOTS,
    FREQUENCY_SONAR_RANGE_M,
)


@dataclass
class GeoPoint:
    """A geographic coordinate with optional depth."""
    latitude: float
    longitude: float
    depth_m: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "depth_m": round(self.depth_m, 1),
        }


@dataclass
class GeotaggedDetection:
    """A detection enriched with geographic coordinates and physical dimensions."""
    detection_id: str
    class_name: str
    class_label: str
    confidence: float
    severity: str
    severity_color: str
    color: str

    # Pixel coordinates
    bbox_px: Dict
    center_px: Dict

    # Geographic coordinates
    location: GeoPoint = None
    estimated_size_m: Dict = field(default_factory=dict)

    # Raw scores
    scores: Dict = field(default_factory=dict)

    # Frame dimensions
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.detection_id,
            "class_name": self.class_name,
            "class_label": self.class_label,
            "confidence": round(self.confidence, 1),
            "severity": self.severity,
            "severity_color": self.severity_color,
            "color": self.color,
            "bbox": self.bbox_px,
            "center_px": self.center_px,
            "location": self.location.to_dict() if self.location else None,
            "estimated_size_m": self.estimated_size_m,
            "scores": self.scores,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


class GeotaggingEngine:
    """
    Converts pixel coordinates to geographic positions using sonar metadata.
    
    Coordinate Mapping Logic:
    - Along-track (Y axis): Pixel row → ping number → distance traveled
    - Cross-track (X axis): Pixel column → slant range → ground range
    
    The engine uses either provided track coordinates or generates
    estimated positions based on configurable defaults.
    """

    def __init__(
        self,
        sonar_range_m: float = DEFAULT_SONAR_RANGE_M,
        ping_interval_m: float = DEFAULT_PING_INTERVAL_M,
        speed_knots: float = DEFAULT_SPEED_KNOTS,
        towfish_altitude_m: float = 0.0,
        layback_m: float = 0.0,
        cable_length_m: Optional[float] = None,
        towfish_depth_m: float = 0.0,
    ):
        self.sonar_range_m = sonar_range_m
        self.ping_interval_m = ping_interval_m
        self.speed_knots = speed_knots
        self.towfish_altitude_m = towfish_altitude_m
        if cable_length_m is not None and cable_length_m > 0:
            self.layback_m = math.sqrt(max(0.0, cable_length_m**2 - towfish_depth_m**2))
        else:
            self.layback_m = layback_m
        self.cable_length_m = cable_length_m
        self.towfish_depth_m = towfish_depth_m
        self.track_points: List[Dict] = []

    @staticmethod
    def slant_to_ground_range(slant_range_m: float, altitude_m: float) -> float:
        """
        Convert slant range to horizontal ground range (T4-C Slant-range correction).
        Rg = sqrt(max(0, Rs^2 - H^2))
        """
        if altitude_m <= 0.0:
            return slant_range_m
        if slant_range_m <= altitude_m:
            return 0.0
        return math.sqrt(slant_range_m * slant_range_m - altitude_m * altitude_m)

    @staticmethod
    def apply_layback(
        vessel_lat: float,
        vessel_lon: float,
        heading_deg: float,
        layback_m: float,
    ) -> Tuple[float, float]:
        """
        Offset vessel GPS backwards along track by layback distance (T4-D Layback correction).
        Towfish trails behind vessel opposite to heading direction.
        """
        if layback_m <= 0.0:
            return vessel_lat, vessel_lon

        heading_rad = math.radians(heading_deg)
        dy = -layback_m * math.cos(heading_rad)
        dx = -layback_m * math.sin(heading_rad)

        fish_lat = vessel_lat + (dy / 111320.0)
        fish_lon = vessel_lon + (dx / (111320.0 * math.cos(math.radians(vessel_lat))))
        return fish_lat, fish_lon

    def load_track_from_csv(self, csv_path: str) -> None:
        """
        Load vessel track coordinates from a CSV file.
        
        Expected columns: latitude, longitude, timestamp (optional), depth (optional)
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Track file not found: {csv_path}")

        self.track_points = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                point = {
                    "lat": float(row.get("latitude", row.get("lat", 0))),
                    "lon": float(row.get("longitude", row.get("lon", row.get("lng", 0)))),
                    "depth": float(row.get("depth", row.get("depth_m", 0))),
                }
                if "timestamp" in row:
                    point["timestamp"] = row["timestamp"]
                self.track_points.append(point)

    def load_track_from_json(self, json_path: str) -> None:
        """Load vessel track from a JSON coordinate file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Track file not found: {json_path}")

        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.track_points = data
        elif "track" in data:
            self.track_points = data["track"]
        elif "waypoints" in data:
            self.track_points = data["waypoints"]

    def set_manual_origin(
        self, lat: float, lon: float, heading_deg: float = 0.0, depth: float = 0.0
    ) -> None:
        """
        Set a manual origin point with heading for simple track estimation.
        Used when no coordinate file is available.
        """
        self.track_points = [
            {
                "lat": lat,
                "lon": lon,
                "heading": heading_deg,
                "depth": depth,
            }
        ]

    def geotag_detections(
        self,
        scored_detections: List,
        image_shape: Tuple[int, int],
        origin_lat: float = 12.9716,
        origin_lon: float = 77.5946,
        heading_deg: float = 0.0,
        towfish_altitude_m: Optional[float] = None,
        layback_m: Optional[float] = None,
        cable_length_m: Optional[float] = None,
        towfish_depth_m: Optional[float] = None,
        frequency: Optional[str] = None,
    ) -> List[GeotaggedDetection]:
        """
        Assign geographic coordinates to all scored detections with slant-range,
        layback, and frequency-aware geometry corrections.
        
        Args:
            scored_detections: List of ScoredDetection objects.
            image_shape: (height, width) of the sonar image.
            origin_lat: Vessel latitude.
            origin_lon: Vessel longitude.
            heading_deg: Vessel heading in degrees (0 = North).
            towfish_altitude_m: Altitude of towfish above seabed (T4-C).
            layback_m: Distance towfish trails vessel in meters (T4-D).
            cable_length_m: Tow cable payout length.
            towfish_depth_m: Depth of towfish below surface.
            frequency: Operating frequency (e.g. '100kHz') for range lookup (Issue 10).
            
        Returns:
            List of GeotaggedDetection objects with lat/lon.
        """
        if frequency and frequency in FREQUENCY_SONAR_RANGE_M:
            self.sonar_range_m = FREQUENCY_SONAR_RANGE_M[frequency]

        altitude = towfish_altitude_m if towfish_altitude_m is not None else self.towfish_altitude_m
        c_len = cable_length_m if cable_length_m is not None else self.cable_length_m
        c_depth = towfish_depth_m if towfish_depth_m is not None else self.towfish_depth_m
        if c_len is not None and c_len > 0:
            eff_layback = math.sqrt(max(0.0, c_len**2 - c_depth**2))
        elif layback_m is not None:
            eff_layback = layback_m
        else:
            eff_layback = self.layback_m

        # Correct vessel origin to towfish origin using layback (T4-D)
        fish_lat, fish_lon = self.apply_layback(origin_lat, origin_lon, heading_deg, eff_layback)

        img_h, img_w = image_shape
        along_track_res = self.ping_interval_m
        cx_img = img_w / 2.0
        r_max = self.sonar_range_m

        geotagged = []
        for idx, sd in enumerate(scored_detections):
            det_id = f"ANM-{idx + 1:03d}"

            # Pixel center & bounds
            bbox = sd.bbox
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            box_w = x2 - x1
            box_h = y2 - y1

            # Slant-range to ground-range mapping for cross-track position (T4-C)
            rs_cx = (abs(cx - cx_img) / max(cx_img, 1.0)) * r_max
            rg_cx = self.slant_to_ground_range(rs_cx, altitude)
            cross_track_m = -rg_cx if cx < cx_img else rg_cx

            # Physical dimensions on seafloor:
            rs_x1 = (abs(x1 - cx_img) / max(cx_img, 1.0)) * r_max
            rs_x2 = (abs(x2 - cx_img) / max(cx_img, 1.0)) * r_max
            rg_x1 = self.slant_to_ground_range(rs_x1, altitude)
            rg_x2 = self.slant_to_ground_range(rs_x2, altitude)
            if (x1 - cx_img) * (x2 - cx_img) >= 0:
                est_width_m = abs(rg_x2 - rg_x1)
            else:
                est_width_m = rg_x1 + rg_x2
            if est_width_m < 0.05 and box_w > 0:
                est_width_m = (box_w / max(img_w, 1)) * (2 * r_max)

            est_height_m = box_h * along_track_res

            # Convert pixel position to geographic coordinates
            if self.track_points and len(self.track_points) > 1:
                geo = self._interpolate_track(cy, cross_track_m, img_h, altitude, eff_layback)
            else:
                geo = self._estimate_position(
                    cy, cross_track_m,
                    fish_lat, fish_lon, heading_deg,
                    along_track_res, altitude,
                )

            geotagged.append(
                GeotaggedDetection(
                    detection_id=det_id,
                    class_name=sd.class_name,
                    class_label=sd.class_label,
                    confidence=sd.final_confidence,
                    severity=sd.severity,
                    severity_color=sd.severity_color,
                    color=sd.color,
                    bbox_px={
                        "x1": bbox[0], "y1": bbox[1],
                        "x2": bbox[2], "y2": bbox[3],
                        "width": box_w, "height": box_h,
                    },
                    center_px={"x": int(cx), "y": int(cy)},
                    location=geo,
                    estimated_size_m={
                        "length_m": round(est_height_m, 2),
                        "width_m": round(est_width_m, 2),
                    },
                    scores=sd.to_dict().get("scores", {}),
                    image_width=img_w,
                    image_height=img_h,
                )
            )

        return geotagged

    def _interpolate_track(
        self,
        row: float,
        cross_track_m: float,
        img_h: int,
        altitude_m: float = 0.0,
        layback_m: float = 0.0,
    ) -> GeoPoint:
        """
        Interpolate position along the loaded vessel track using slant-range corrected cross_track_m
        and layback correction for towfish offset.
        """
        n = len(self.track_points)
        progress = row / max(img_h, 1)  # 0.0 to 1.0
        idx_float = progress * (n - 1)
        idx = min(int(idx_float), n - 2)
        frac = idx_float - idx

        p1 = self.track_points[idx]
        p2 = self.track_points[idx + 1]

        # Linear interpolation along vessel track
        vessel_lat = p1["lat"] + frac * (p2["lat"] - p1["lat"])
        vessel_lon = p1["lon"] + frac * (p2["lon"] - p1["lon"])
        depth = p1.get("depth", altitude_m) + frac * (p2.get("depth", altitude_m) - p1.get("depth", altitude_m))

        # Calculate track heading between p1 and p2 (radians from North)
        dlat_m = (p2["lat"] - p1["lat"]) * 111320.0
        dlon_m = (p2["lon"] - p1["lon"]) * 111320.0 * math.cos(math.radians(vessel_lat))
        heading_rad = math.atan2(dlon_m, dlat_m)
        heading_deg = math.degrees(heading_rad) % 360.0

        # Offset vessel GPS backwards along track by layback distance (T4-D)
        fish_lat, fish_lon = self.apply_layback(vessel_lat, vessel_lon, heading_deg, layback_m)

        # Perpendicular offset from towfish position using slant-range corrected cross-track distance
        dy_cross = -cross_track_m * math.sin(heading_rad)
        dx_cross = cross_track_m * math.cos(heading_rad)

        lat_final = fish_lat + (dy_cross / 111320.0)
        lon_final = fish_lon + (dx_cross / (111320.0 * math.cos(math.radians(fish_lat))))

        return GeoPoint(
            latitude=lat_final,
            longitude=lon_final,
            depth_m=depth,
        )

    def _estimate_position(
        self,
        row: float,
        cross_track_m: float,
        origin_lat: float,
        origin_lon: float,
        heading_deg: float,
        along_track_res: float,
        altitude_m: float = 0.0,
    ) -> GeoPoint:
        """
        Estimate geographic position from a single origin point and heading
        using slant-range corrected cross_track_m.
        """
        heading_rad = math.radians(heading_deg)

        # Along-track distance from top of image
        along_track_m = row * along_track_res

        # Convert to lat/lon offsets
        # Along-track: in heading direction; Cross-track: perpendicular
        dy = along_track_m * math.cos(heading_rad) - cross_track_m * math.sin(heading_rad)
        dx = along_track_m * math.sin(heading_rad) + cross_track_m * math.cos(heading_rad)

        lat = origin_lat + (dy / 111320.0)
        lon = origin_lon + (dx / (111320.0 * math.cos(math.radians(origin_lat))))

        return GeoPoint(
            latitude=lat,
            longitude=lon,
            depth_m=altitude_m,
        )
