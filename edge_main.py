"""
Edge Embedded Entry Point
===========================
Standalone launcher for the embedded SSS detection system.

Starts three components together:
  1. SonarReader         — reads pings from hardware/file
  2. RealtimeEngine      — detection + WebSocket broadcast
  3. FastAPI server      — live dashboard + REST API + WebSocket endpoint

Usage:
    # Test with file replay (development)
    python edge_main.py --interface file --source uploads/

    # Serial SSS hardware (RS-232)
    python edge_main.py --interface serial --port COM3 --baud 115200

    # UDP SSS hardware
    python edge_main.py --interface udp --udp-port 4000

    # XTF file replay
    python edge_main.py --interface xtf --source survey.xtf --loop

    # Auto-detect hardware platform
    python edge_main.py --interface file --source uploads/ --platform auto
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import STATIC_DIR, TEMPLATES_DIR, UPLOADS_DIR
from embedded.hardware_config import HardwarePlatform, get_profile
from embedded.sonar_reader import create_reader
from embedded.realtime_engine import RealtimeEngine
from embedded.database import DetectionDatabase
from api.routes import router as api_router

logger = logging.getLogger("edge_main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Global State ─────────────────────────────────────────────────────────────

_engine: RealtimeEngine = None
_db: DetectionDatabase = None
_profile = None


# ─── WebSocket Manager ────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages live WebSocket connections from dashboard clients."""

    def __init__(self):
        self._connections: dict[WebSocket, asyncio.Queue] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._connections[ws] = q
        if _engine:
            _engine.subscribe(q)
        logger.info(f"WS client connected ({len(self._connections)} total)")
        return q

    def disconnect(self, ws: WebSocket):
        q = self._connections.pop(ws, None)
        if q and _engine:
            _engine.unsubscribe(q)
        logger.info(f"WS client disconnected ({len(self._connections)} remaining)")

    def broadcast_sync(self, data: dict):
        """Called from background thread — puts to all queues."""
        dead = []
        for ws, q in self._connections.items():
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.pop(ws, None)


manager = ConnectionManager()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: engine is already running before FastAPI starts."""
    logger.info("Edge API server starting up")
    STATIC_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
    yield
    logger.info("Edge API server shutting down")


edge_app = FastAPI(
    title="SSS Real-Time Detection Edge API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

edge_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount existing REST API routes
edge_app.include_router(api_router)

# Static files
if STATIC_DIR.exists():
    edge_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
edge_app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ─── Edge-Specific Endpoints ──────────────────────────────────────────────────

@edge_app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the live sonar dashboard."""
    index = TEMPLATES_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found — build templates/index.html</h1>")


@edge_app.get("/edge/status")
async def edge_status():
    """Current engine status and session info."""
    if _engine is None:
        return {"status": "not_started"}
    stats = _engine.stats
    db_stats = _db.get_stats(_engine.session_id) if _db else {}
    return {
        **stats,
        "platform": _profile.name if _profile else "unknown",
        "db_stats": db_stats,
    }


@edge_app.post("/edge/start")
async def start_session():
    """Manually start a new detection session."""
    if _engine and _engine.is_running:
        return {"status": "already_running", "session_id": _engine.session_id}
    if _engine:
        _engine.start()
        return {"status": "started", "session_id": _engine.session_id}
    return {"status": "error", "detail": "Engine not initialized"}


@edge_app.post("/edge/stop")
async def stop_session():
    """Stop the current detection session."""
    if _engine and _engine.is_running:
        sid = _engine.session_id
        _engine.stop()
        return {"status": "stopped", "session_id": sid}
    return {"status": "not_running"}


@edge_app.get("/edge/sessions")
async def list_sessions():
    """List all past detection sessions."""
    if _db is None:
        return {"sessions": []}
    return {"sessions": _db.list_sessions()}


@edge_app.get("/edge/sessions/{session_id}/detections")
async def get_session_detections(session_id: str):
    """Get all detections for a session."""
    if _db is None:
        return {"detections": []}
    return {"detections": _db.get_session_detections(session_id)}


@edge_app.get("/edge/sessions/{session_id}/export.csv")
async def export_csv(session_id: str):
    """Export session detections as CSV."""
    if _db is None:
        return JSONResponse({"error": "DB not initialized"}, status_code=500)
    csv_content = _db.export_csv(session_id)
    from fastapi.responses import Response
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.csv"},
    )


@edge_app.get("/edge/sessions/{session_id}/export.geojson")
async def export_geojson(session_id: str):
    """Export session detections as GeoJSON."""
    if _db is None:
        return JSONResponse({"error": "DB not initialized"}, status_code=500)
    return _db.export_geojson(session_id)


# ─── WebSocket Live Stream ────────────────────────────────────────────────────

@edge_app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket endpoint for real-time detection stream.

    Dashboard clients connect here to receive live FrameResult events
    as JSON messages. Each message contains:
      - annotated sonar frame (base64 JPEG)
      - list of detections with GPS and severity
      - session statistics
    """
    q = await manager.connect(ws)
    try:
        # Send current status immediately on connect
        if _engine:
            await ws.send_text(json.dumps({
                "type": "connected",
                "session_id": _engine.session_id,
                "stats": _engine.stats,
            }))

        while True:
            try:
                # Wait for next frame result from engine (with timeout)
                data = await asyncio.wait_for(q.get(), timeout=30.0)
                await ws.send_text(json.dumps(data, default=str))
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await ws.send_text(json.dumps({
                    "type": "heartbeat",
                    "stats": _engine.stats if _engine else {},
                }))
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SSS Real-Time Detection Edge System")
    p.add_argument("--interface", choices=["serial", "udp", "xtf", "file"],
                   default="file", help="Sonar hardware interface type")
    p.add_argument("--source", default="uploads/",
                   help="File path (file/xtf interface) or directory")
    p.add_argument("--port", default="COM3",
                   help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=115200,
                   help="Serial baud rate")
    p.add_argument("--udp-port", type=int, default=4000,
                   help="UDP listen port")
    p.add_argument("--udp-host", default="0.0.0.0",
                   help="UDP listen host")
    p.add_argument("--loop", action="store_true",
                   help="Loop file/XTF playback")
    p.add_argument("--fps", type=float, default=20.0,
                   help="File reader replay FPS (pings/sec)")
    p.add_argument("--platform", default="auto",
                   choices=["auto", "jetson_orin", "jetson_nano", "rpi5", "rpi4",
                            "windows_pc", "linux_pc"],
                   help="Hardware platform profile")
    p.add_argument("--host", default="0.0.0.0", help="API server host")
    p.add_argument("--api-port", type=int, default=8000, help="API server port")
    p.add_argument("--db", default="data/detections.db",
                   help="SQLite database path")
    p.add_argument("--no-autostart", action="store_true",
                   help="Don't auto-start detection (start manually via /edge/start)")
    return p.parse_args()


def main():
    global _engine, _db, _profile
    args = parse_args()

    # ── Load hardware profile ─────────────────────────────────────────
    platform = HardwarePlatform(args.platform) if args.platform != "auto" else HardwarePlatform.AUTO
    _profile = get_profile(platform)
    logger.info(f"Hardware profile: {_profile.name}")

    # ── Initialize database ───────────────────────────────────────────
    _db = DetectionDatabase(db_path=args.db)
    logger.info(f"Database: {args.db}")

    # ── Create sonar reader ───────────────────────────────────────────
    reader_kwargs = {}
    if args.interface == "file":
        reader_kwargs = {"paths": args.source, "fps": args.fps, "loop": args.loop}
    elif args.interface == "serial":
        reader_kwargs = {"port": args.port, "baud_rate": args.baud}
    elif args.interface == "udp":
        reader_kwargs = {"host": args.udp_host, "port": args.udp_port}
    elif args.interface == "xtf":
        reader_kwargs = {"path": args.source, "loop": args.loop}

    reader = create_reader(args.interface, **reader_kwargs)
    logger.info(f"Sonar reader: {reader.__class__.__name__}")

    # ── Create detection engine ───────────────────────────────────────
    _engine = RealtimeEngine(reader=reader, profile=_profile, db=_db)

    # ── Auto-start detection ──────────────────────────────────────────
    if not args.no_autostart:
        session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"
        _engine.start(session_id=session_id)
        logger.info(f"Detection session started: {session_id}")

    # ── Graceful shutdown ─────────────────────────────────────────────
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received")
        if _engine:
            _engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start API server ──────────────────────────────────────────────
    logger.info(f"Starting edge server on http://{args.host}:{args.api_port}")
    logger.info(f"Dashboard: http://localhost:{args.api_port}/")
    logger.info(f"Live WS:   ws://localhost:{args.api_port}/ws/live")
    logger.info(f"API docs:  http://localhost:{args.api_port}/docs")

    uvicorn.run(
        edge_app,
        host=args.host,
        port=args.api_port,
        log_level="warning",  # Keep uvicorn quiet; our logger is verbose
    )


if __name__ == "__main__":
    main()
