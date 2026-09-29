"""
SQLite Detection Database
==========================
Persistent storage for all sonar detection sessions and individual
detection events. Survives server restarts.

Schema:
  sessions   — one row per sonar survey session
  detections — one row per detected anomaly
  pings      — optional ping statistics per session

Exports to CSV and GeoJSON for GIS integration.
"""

import csv
import json
import sqlite3
import threading
import time
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    session_id: str
    started_at: float
    ended_at: Optional[float] = None
    total_pings: int = 0
    total_detections: int = 0
    source_interface: str = ""
    hardware_platform: str = ""
    notes: str = ""


@dataclass
class DetectionRecord:
    session_id: str
    detection_id: str
    timestamp: float
    class_name: str
    class_label: str
    confidence: float
    severity: str
    latitude: Optional[float]
    longitude: Optional[float]
    depth_m: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    est_length_m: float
    est_width_m: float
    yolo_score: float
    shadow_score: float
    morph_score: float
    texture_score: float
    frame_number: int
    annotated_image_path: str = ""


# ─── Database Manager ─────────────────────────────────────────────────────────

class DetectionDatabase:
    """
    Thread-safe SQLite database for detection persistence.

    Uses WAL mode for concurrent read/write performance.
    All writes are serialized through a lock to prevent corruption.
    """

    def __init__(self, db_path: str = "data/detections.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id         TEXT PRIMARY KEY,
                    started_at         REAL NOT NULL,
                    ended_at           REAL,
                    total_pings        INTEGER DEFAULT 0,
                    total_detections   INTEGER DEFAULT 0,
                    source_interface   TEXT DEFAULT '',
                    hardware_platform  TEXT DEFAULT '',
                    notes              TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id         TEXT NOT NULL,
                    detection_id       TEXT NOT NULL,
                    timestamp          REAL NOT NULL,
                    class_name         TEXT NOT NULL,
                    class_label        TEXT NOT NULL,
                    confidence         REAL NOT NULL,
                    severity           TEXT NOT NULL,
                    latitude           REAL,
                    longitude          REAL,
                    depth_m            REAL DEFAULT 0,
                    bbox_x1            INTEGER DEFAULT 0,
                    bbox_y1            INTEGER DEFAULT 0,
                    bbox_x2            INTEGER DEFAULT 0,
                    bbox_y2            INTEGER DEFAULT 0,
                    est_length_m       REAL DEFAULT 0,
                    est_width_m        REAL DEFAULT 0,
                    yolo_score         REAL DEFAULT 0,
                    shadow_score       REAL DEFAULT 0,
                    morph_score        REAL DEFAULT 0,
                    texture_score      REAL DEFAULT 0,
                    frame_number       INTEGER DEFAULT 0,
                    annotated_path     TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_det_session
                    ON detections(session_id);
                CREATE INDEX IF NOT EXISTS idx_det_timestamp
                    ON detections(timestamp);
                CREATE INDEX IF NOT EXISTS idx_det_severity
                    ON detections(severity);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Session Management ─────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        interface: str = "",
        platform: str = "",
        notes: str = "",
    ) -> SessionRecord:
        """Create and persist a new survey session."""
        rec = SessionRecord(
            session_id=session_id,
            started_at=time.time(),
            source_interface=interface,
            hardware_platform=platform,
            notes=notes,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, started_at, source_interface, hardware_platform, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (rec.session_id, rec.started_at, rec.source_interface,
                 rec.hardware_platform, rec.notes),
            )
        return rec

    def end_session(self, session_id: str, total_pings: int, total_detections: int):
        """Mark a session as complete."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE sessions
                   SET ended_at=?, total_pings=?, total_detections=?
                   WHERE session_id=?""",
                (time.time(), total_pings, total_detections, session_id),
            )

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Detection Logging ─────────────────────────────────────────────

    def log_detection(self, det: DetectionRecord):
        """Insert one detection record."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO detections
                   (session_id, detection_id, timestamp, class_name, class_label,
                    confidence, severity, latitude, longitude, depth_m,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    est_length_m, est_width_m,
                    yolo_score, shadow_score, morph_score, texture_score,
                    frame_number, annotated_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (det.session_id, det.detection_id, det.timestamp,
                 det.class_name, det.class_label, det.confidence, det.severity,
                 det.latitude, det.longitude, det.depth_m,
                 det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2,
                 det.est_length_m, det.est_width_m,
                 det.yolo_score, det.shadow_score, det.morph_score, det.texture_score,
                 det.frame_number, det.annotated_image_path),
            )

    def get_session_detections(self, session_id: str) -> List[Dict]:
        """Get all detections for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE session_id=? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        """Get the most recent detections across all sessions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, session_id: Optional[str] = None) -> Dict:
        """Compute detection statistics."""
        where = "WHERE session_id=?" if session_id else ""
        params = (session_id,) if session_id else ()
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM detections {where}", params
            ).fetchone()[0]
            by_severity = {}
            for row in conn.execute(
                f"SELECT severity, COUNT(*) as cnt FROM detections {where} GROUP BY severity",
                params,
            ):
                by_severity[row["severity"]] = row["cnt"]
            by_class = {}
            for row in conn.execute(
                f"SELECT class_name, COUNT(*) as cnt FROM detections {where} GROUP BY class_name",
                params,
            ):
                by_class[row["class_name"]] = row["cnt"]
        return {
            "total": total,
            "by_severity": by_severity,
            "by_class": by_class,
        }

    # ── Export ────────────────────────────────────────────────────────

    def export_csv(self, session_id: Optional[str] = None) -> str:
        """Export detections as CSV string."""
        detections = (
            self.get_session_detections(session_id)
            if session_id
            else self.get_recent_detections(limit=10000)
        )
        if not detections:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=detections[0].keys())
        writer.writeheader()
        writer.writerows(detections)
        return output.getvalue()

    def export_geojson(self, session_id: Optional[str] = None) -> Dict:
        """Export detections with GPS as GeoJSON FeatureCollection."""
        detections = (
            self.get_session_detections(session_id)
            if session_id
            else self.get_recent_detections(limit=10000)
        )
        features = []
        for d in detections:
            if d.get("latitude") and d.get("longitude"):
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [d["longitude"], d["latitude"]],
                    },
                    "properties": {
                        "id": d["detection_id"],
                        "class": d["class_name"],
                        "label": d["class_label"],
                        "confidence": round(d["confidence"], 1),
                        "severity": d["severity"],
                        "timestamp": datetime.fromtimestamp(
                            d["timestamp"], tz=timezone.utc
                        ).isoformat(),
                        "est_length_m": d.get("est_length_m", 0),
                        "est_width_m": d.get("est_width_m", 0),
                    },
                })
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "session_id": session_id,
                "total_features": len(features),
                "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        }

    def delete_session(self, session_id: str):
        """Delete a session and all its detections."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM detections WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

    def checkpoint_wal(self):
        """Force a WAL checkpoint to truncate log and reclaim disk space (T3-C)."""
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    def enforce_storage_limit(self, max_storage_gb: float, data_dir: Path):
        """
        Prune oldest image files and database records when storage exceeds max_storage_gb (T3-D).
        """
        if max_storage_gb <= 0 or not data_dir.exists():
            return

        max_bytes = max_storage_gb * 1024 * 1024 * 1024
        try:
            total_bytes = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
            if total_bytes <= max_bytes:
                return

            # Prune oldest images first
            img_dir = data_dir / "images"
            if img_dir.exists():
                files = sorted(img_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
                for f in files:
                    try:
                        sz = f.stat().st_size
                        f.unlink()
                        total_bytes -= sz
                        if total_bytes <= max_bytes * 0.90:  # prune to 90% headroom
                            break
                    except Exception:
                        pass

            self.checkpoint_wal()
        except Exception:
            pass
