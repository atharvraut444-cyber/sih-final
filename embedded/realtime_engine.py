"""
Real-Time SSS Detection Engine
================================
Core processing loop for embedded deployment.

Reads sonar pings → assembles frames → runs AI pipeline → broadcasts results.

Architecture:
  SonarReader (background thread)
      ↓ pings (queue)
  FrameAssembler (accumulates pings_per_frame rows)
      ↓ frame image (numpy array)
  DetectionPipeline (existing core/ modules)
      ↓ geotagged detections
  WebSocket broadcaster → connected dashboard clients
  DetectionDatabase → SQLite persistence
  GPIO alerter → LED on Jetson/RPi (if available)
"""

import asyncio
import logging
import math
import threading
import time
import uuid
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessor import SonarPreprocessor
from core.detector import SonarDetector
from core.confidence import ConfidenceScorer
from core.geotagging import GeotaggingEngine
from embedded.sonar_reader import SonarReader, SonarPing
from embedded.database import DetectionDatabase, DetectionRecord, SessionRecord
from embedded.hardware_config import HardwareProfile

logger = logging.getLogger(__name__)


# ─── Frame Event ──────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Result of processing one sonar frame — sent to WebSocket clients."""
    session_id: str
    frame_number: int
    timestamp: float
    ping_range: tuple          # (first_ping, last_ping)
    image_shape: tuple
    detections: List[Dict]
    summary: Dict
    annotated_image_b64: Optional[str] = None   # Base64 JPEG for live preview
    latest_gps: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "type": "frame_result",
            "session_id": self.session_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "ping_range": list(self.ping_range),
            "image_shape": list(self.image_shape),
            "detections": self.detections,
            "summary": self.summary,
            "annotated_image": self.annotated_image_b64,
            "gps": self.latest_gps,
        }


# ─── Frame Assembler ──────────────────────────────────────────────────────────

class FrameAssembler:
    """
    Accumulates sonar pings into a 2D image frame.

    Each ping is one horizontal row. Once `pings_per_frame` rows are
    accumulated, the frame is released for detection.

    Also maintains a rolling waterfall buffer for the live display.
    """

    def __init__(self, pings_per_frame: int = 512, samples_per_ping: int = 1000):
        self.pings_per_frame = pings_per_frame
        self.samples_per_ping = samples_per_ping
        self._rows: List[np.ndarray] = []
        self._ping_meta: List[SonarPing] = []

    def add_ping(self, ping: SonarPing) -> Optional[np.ndarray]:
        """
        Add a ping row. Returns a complete frame when enough pings accumulated,
        otherwise returns None.
        """
        row = ping.combined
        # Resize to consistent width
        if len(row) != self.samples_per_ping:
            row = cv2.resize(row.reshape(1, -1), (self.samples_per_ping, 1)).flatten()
        self._rows.append(row.astype(np.uint8))
        self._ping_meta.append(ping)

        if len(self._rows) >= self.pings_per_frame:
            frame = np.vstack(self._rows[:self.pings_per_frame])
            meta = self._ping_meta[:self.pings_per_frame]
            # Slide window — keep 25% overlap for continuity
            overlap = self.pings_per_frame // 4
            self._rows = self._rows[self.pings_per_frame - overlap:]
            self._ping_meta = self._ping_meta[self.pings_per_frame - overlap:]
            return frame, meta
        return None, None


# ─── Realtime Engine ──────────────────────────────────────────────────────────

class RealtimeEngine:
    """
    Main real-time detection engine.

    Lifecycle:
      engine.start(session_id)   — begins reading + processing
      engine.stop()              — graceful shutdown
      engine.subscribe(ws_queue) — add a WebSocket client to receive results
    """

    def __init__(
        self,
        reader: SonarReader,
        profile: HardwareProfile,
        db: DetectionDatabase,
    ):
        self.reader = reader
        self.profile = profile
        self.db = db

        # Core AI modules
        self.preprocessor = SonarPreprocessor()
        self.detector = SonarDetector()
        self.scorer = ConfidenceScorer()
        self.geo_engine = GeotaggingEngine(sonar_range_m=profile.sonar_range_m)

        # Frame assembly
        self.assembler = FrameAssembler(
            pings_per_frame=profile.pings_per_frame,
            samples_per_ping=1000,
        )

        # State
        self._running = False
        self._session_id: Optional[str] = None
        self._frame_number = 0
        self._total_pings = 0
        self._total_detections = 0
        self._processing_thread: Optional[threading.Thread] = None

        # WebSocket subscriber queues
        self._subscribers: Set[asyncio.Queue] = set()
        self._subscribers_lock = threading.Lock()

        # Cross-frame deduplication tracking (T2-C)
        self._recent_detections: List[Dict] = []
        self._recent_detections_lock = threading.Lock()

        # Decoupled WebSocket broadcast queue and worker (T3-B)
        self._broadcast_queue: Queue = Queue(maxsize=32)
        self._broadcast_thread: Optional[threading.Thread] = None

        # Serialized GPIO alert queue and worker (T3-E)
        self._alert_queue: Queue = Queue(maxsize=32)
        self._alert_thread: Optional[threading.Thread] = None

        # GPIO
        self._gpio = None
        self._init_gpio()

        logger.info(
            f"RealtimeEngine initialized — "
            f"platform: {profile.platform} | "
            f"model: {self.detector.model_name} | "
            f"demo_mode: {self.detector.demo_mode}"
        )

    # ── GPIO Alert LED ────────────────────────────────────────────────

    def _init_gpio(self):
        """Initialize GPIO for alert LED (Jetson/RPi only)."""
        if not self.profile.gpio_available:
            return
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            if self.profile.alert_led_pin:
                GPIO.setup(self.profile.alert_led_pin, GPIO.OUT)
            if self.profile.status_led_pin:
                GPIO.setup(self.profile.status_led_pin, GPIO.OUT)
                GPIO.output(self.profile.status_led_pin, GPIO.HIGH)
            self._gpio = GPIO
            logger.info("GPIO initialized (RPi)")
        except ImportError:
            try:
                import Jetson.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                if self.profile.alert_led_pin:
                    GPIO.setup(self.profile.alert_led_pin, GPIO.OUT)
                if self.profile.status_led_pin:
                    GPIO.setup(self.profile.status_led_pin, GPIO.OUT)
                    GPIO.output(self.profile.status_led_pin, GPIO.HIGH)
                self._gpio = GPIO
                logger.info("GPIO initialized (Jetson)")
            except ImportError:
                logger.debug("GPIO not available on this platform")

    def _alert_led(self, severity: str):
        """Pulse alert LED for critical detections."""
        if not self._gpio or not self.profile.alert_led_pin:
            return
        if severity == "critical":
            pin = self.profile.alert_led_pin
            for _ in range(3):
                self._gpio.output(pin, self._gpio.HIGH)
                time.sleep(0.1)
                self._gpio.output(pin, self._gpio.LOW)
    # ── Cross-Frame Deduplication (T2-C) ──────────────────────────────

    def _is_duplicate_detection(
        self, geo, max_dist_m: float = 5.0, window_sec: float = 30.0
    ) -> bool:
        """
        Check if a detection matches a recently logged detection within spatial/temporal window (T2-C).
        Prevents duplicate database entries across overlapping consecutive frames.
        """
        if not geo.location or geo.location.latitude is None or geo.location.longitude is None:
            return False

        lat, lon = geo.location.latitude, geo.location.longitude
        cls = geo.class_name
        now = time.time()

        with self._recent_detections_lock:
            # Prune records older than window_sec
            self._recent_detections = [
                d for d in self._recent_detections if (now - d["timestamp"]) < window_sec
            ]

            for past in self._recent_detections:
                if past["class_name"] == cls:
                    dlat = (lat - past["lat"]) * 111320.0
                    dlon = (lon - past["lon"]) * 111320.0 * math.cos(math.radians(lat))
                    dist = math.hypot(dlat, dlon)
                    if dist < max_dist_m:
                        past["timestamp"] = now
                        return True

            self._recent_detections.append({
                "lat": lat,
                "lon": lon,
                "class_name": cls,
                "timestamp": now,
                "detection_id": geo.detection_id,
            })
            return False

    # ── Session Control ───────────────────────────────────────────────

    def start(self, session_id: Optional[str] = None):
        """Start the engine and sonar reader."""
        self._session_id = session_id or f"SES-{uuid.uuid4().hex[:8].upper()}"
        self._frame_number = 0
        self._total_pings = 0
        self._total_detections = 0
        self._running = True

        # Create DB session
        self.db.create_session(
            session_id=self._session_id,
            interface=self.reader.__class__.__name__,
            platform=self.profile.platform.value,
        )

        # Start sonar reader
        self.reader.start()

        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="RealtimeEngine",
            daemon=True,
        )
        self._processing_thread.start()

        # Start decoupled broadcast and alert workers (T3-B, T3-E)
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_worker,
            name="BroadcastWorker",
            daemon=True,
        )
        self._broadcast_thread.start()

        self._alert_thread = threading.Thread(
            target=self._alert_worker,
            name="AlertWorker",
            daemon=True,
        )
        self._alert_thread.start()

        logger.info(f"RealtimeEngine started — session: {self._session_id}")

    def stop(self):
        """Gracefully stop the engine."""
        self._running = False
        self.reader.stop()

        try:
            self._broadcast_queue.put_nowait(None)
            self._alert_queue.put_nowait(None)
        except Exception:
            pass

        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

        # Finalize DB session
        if self._session_id:
            self.db.end_session(
                self._session_id,
                self._total_pings,
                self._total_detections,
            )

        # Turn off LEDs
        if self._gpio:
            if self.profile.status_led_pin:
                self._gpio.output(self.profile.status_led_pin, self._gpio.LOW)
            self._gpio.cleanup()

        logger.info(
            f"RealtimeEngine stopped — "
            f"session: {self._session_id} | "
            f"pings: {self._total_pings} | "
            f"detections: {self._total_detections}"
        )

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict:
        return {
            "session_id": self._session_id,
            "running": self._running,
            "total_pings": self._total_pings,
            "total_frames": self._frame_number,
            "total_detections": self._total_detections,
            "model": self.detector.model_name,
            "demo_mode": self.detector.demo_mode,
        }

    # ── WebSocket Subscriptions ───────────────────────────────────────

    def subscribe(self, queue: asyncio.Queue):
        """Register an asyncio queue to receive FrameResult dicts."""
        with self._subscribers_lock:
            self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        with self._subscribers_lock:
            self._subscribers.discard(queue)

    def _broadcast(self, data: Dict):
        """Enqueue data for WebSocket subscriber broadcast without blocking detection thread (T3-B)."""
        try:
            self._broadcast_queue.put_nowait(data)
        except Exception:
            pass

    def _broadcast_worker(self):
        """Worker thread to dispatch frames to WebSocket clients off detection thread (T3-B)."""
        while self._running:
            try:
                data = self._broadcast_queue.get(timeout=0.5)
                if data is None:
                    break
                with self._subscribers_lock:
                    dead = set()
                    for q in self._subscribers:
                        try:
                            q.put_nowait(data)
                        except Exception:
                            dead.add(q)
                    self._subscribers -= dead
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Broadcast worker error: {e}")

    def _alert_worker(self):
        """Worker thread to serialize GPIO alert pulses without thread storm (T3-E)."""
        while self._running:
            try:
                severity = self._alert_queue.get(timeout=0.5)
                if severity is None:
                    break
                self._alert_led(severity)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Alert worker error: {e}")

    # ── Processing Loop ───────────────────────────────────────────────

    def _processing_loop(self):
        """Main loop: ping → frame → detect → broadcast."""
        logger.info("Processing loop started")
        frame_interval = 1.0 / max(self.profile.target_fps, 1.0)
        last_frame_time = 0.0

        while self._running:
            ping = self.reader.get_ping(timeout=1.0)
            if ping is None:
                if not self.reader._running:
                    logger.info("Sonar reader stopped — ending processing loop")
                    break
                continue

            self._total_pings += 1

            # Assemble frame
            result = self.assembler.add_ping(ping)
            if result[0] is None:
                continue
            frame_image, frame_pings = result

            # Throttle to target FPS
            now = time.time()
            if (now - last_frame_time) < frame_interval:
                continue
            last_frame_time = now

            # Run detection pipeline
            try:
                frame_result = self._process_frame(frame_image, frame_pings)
                self._frame_number += 1
                self._total_detections += len(frame_result.detections)

                # GPIO alert: enqueue alert to serialized dispatcher (T3-E)
                for det in frame_result.detections:
                    if det.get("severity") == "critical":
                        try:
                            self._alert_queue.put_nowait("critical")
                        except Exception:
                            pass
                        break

                # Broadcast to WebSocket clients (decoupled via queue T3-B)
                self._broadcast(frame_result.to_dict())

                # Periodic storage management and WAL checkpointing (T3-C, T3-D)
                if self._frame_number % 50 == 0:
                    try:
                        self.db.enforce_storage_limit(
                            getattr(self.profile, "max_storage_gb", 16.0),
                            Path(self.profile.data_dir),
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.exception(f"Frame processing error: {e}")

        logger.info("Processing loop ended")

    def _process_frame(self, frame: np.ndarray, pings: List[SonarPing]) -> FrameResult:
        """Run the full AI pipeline on one sonar frame."""
        t0 = time.time()
        first_ping = pings[0] if pings else None
        last_ping = pings[-1] if pings else None

        # ── Preprocess ────────────────────────────────────────────────
        prep = self.preprocessor.process_array(frame)

        # ── Detect ────────────────────────────────────────────────────
        det_result = self.detector.detect_tiled(
            tiles=prep.tiles,
            tile_positions=prep.tile_positions,
            full_image_shape=prep.processed.shape[:2],
            confidence_threshold=self.profile.confidence_threshold,
            iou_threshold=self.profile.iou_threshold,
            input_size=self.profile.model_input_size,
        )

        # ── Score & Filter ────────────────────────────────────────────
        scored = self.scorer.score_detections(
            detections=det_result.detections,
            image=prep.processed,
            shadow_map=prep.shadow_map,
        )
        filtered = self.scorer.filter_detections(
            scored,
            min_confidence=self.profile.confidence_threshold * 100.0,
        )

        # ── Geotag (T4-C Slant Range, T4-D Layback, Issue 10 Range) ───
        origin_lat = first_ping.latitude if (first_ping and first_ping.latitude is not None) else 12.9716
        origin_lon = first_ping.longitude if (first_ping and first_ping.longitude is not None) else 77.5946
        heading    = first_ping.heading_deg if first_ping else 0.0
        altitude   = first_ping.altitude_m if first_ping else 0.0

        geotagged = self.geo_engine.geotag_detections(
            scored_detections=filtered,
            image_shape=prep.processed.shape[:2],
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            heading_deg=heading,
            towfish_altitude_m=altitude,
            layback_m=getattr(self.profile, "layback_m", 0.0),
            cable_length_m=getattr(self.profile, "cable_length_m", None),
            towfish_depth_m=getattr(self.profile, "towfish_depth_m", 0.0),
        )

        # ── Save annotated frame ──────────────────────────────────────
        annotated = self._draw_detections(prep.processed, filtered)
        b64_img = self._encode_b64(annotated)

        # Optionally save to disk
        frame_path = ""
        if geotagged:
            img_dir = Path(self.profile.data_dir) / "images"
            img_dir.mkdir(exist_ok=True)
            frame_path = str(img_dir / f"{self._session_id}_f{self._frame_number:05d}.jpg")
            cv2.imwrite(frame_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # ── Persist to DB (with cross-frame deduplication T2-C) ─────────
        for i, geo in enumerate(geotagged):
            if self._is_duplicate_detection(geo):
                continue
            det_dict = geo.to_dict()
            scores = det_dict.get("scores", {})
            self.db.log_detection(DetectionRecord(
                session_id=self._session_id,
                detection_id=geo.detection_id,
                timestamp=time.time(),
                class_name=geo.class_name,
                class_label=geo.class_label,
                confidence=geo.confidence,
                severity=geo.severity,
                latitude=geo.location.latitude if geo.location else None,
                longitude=geo.location.longitude if geo.location else None,
                depth_m=geo.location.depth_m if geo.location else 0.0,
                bbox_x1=geo.bbox_px.get("x1", 0),
                bbox_y1=geo.bbox_px.get("y1", 0),
                bbox_x2=geo.bbox_px.get("x2", 0),
                bbox_y2=geo.bbox_px.get("y2", 0),
                est_length_m=geo.estimated_size_m.get("length_m", 0),
                est_width_m=geo.estimated_size_m.get("width_m", 0),
                yolo_score=scores.get("yolo_confidence", 0),
                shadow_score=scores.get("shadow_score", 0),
                morph_score=scores.get("morphology_score", 0),
                texture_score=scores.get("texture_score", 0),
                frame_number=self._frame_number,
                annotated_image_path=frame_path,
            ))

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            f"[{self._session_id}] Frame {self._frame_number}: "
            f"{len(geotagged)} detections in {elapsed_ms:.1f}ms"
        )

        # ── Build GPS info ─────────────────────────────────────────────
        gps = None
        if first_ping and first_ping.latitude:
            gps = {
                "lat": first_ping.latitude,
                "lon": first_ping.longitude,
                "heading": first_ping.heading_deg,
                "speed_knots": first_ping.speed_knots,
            }

        return FrameResult(
            session_id=self._session_id,
            frame_number=self._frame_number,
            timestamp=time.time(),
            ping_range=(
                pings[0].ping_number if pings else 0,
                pings[-1].ping_number if pings else 0,
            ),
            image_shape=frame.shape[:2],
            detections=[geo.to_dict() for geo in geotagged],
            summary=self.scorer.get_statistics(filtered),
            annotated_image_b64=b64_img,
            latest_gps=gps,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _draw_detections(self, image: np.ndarray, scored_detections) -> np.ndarray:
        """Draw bounding boxes colored by severity."""
        if len(image.shape) == 2:
            out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            out = image.copy()

        COLORS = {
            "critical":   (57, 71, 255),
            "moderate":   (2, 165, 255),
            "low":        (115, 213, 46),
            "negligible": (140, 125, 116),
        }
        for det in scored_detections:
            x1, y1, x2, y2 = det.bbox
            bgr = COLORS.get(det.severity, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
            label = f"{det.class_name} {det.final_confidence:.0f}%"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(out, (x1, max(0, y1 - lh - bl - 3)), (x1 + lw + 4, y1), bgr, -1)
            cv2.putText(out, label, (x1 + 2, y1 - bl - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _encode_b64(self, image: np.ndarray) -> str:
        """Encode image as base64 JPEG for WebSocket transmission."""
        import base64
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
