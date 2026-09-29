"""
SENORITA — Side-Scan Sonar Marine Debris Detection System
===========================================================
Entry point for the REST API server.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or directly:
    python main.py
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    API_HOST,
    API_PORT,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
)
from api.routes import router
from api.pipeline import get_pipeline


# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sonar_api")


# ─── Application Lifespan (startup / shutdown) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load the YOLOv8 model so the first request isn't slow.
    Shutdown: nothing special needed for now.
    """
    logger.info("=" * 60)
    logger.info("  SENORITA — SSS Marine Debris Detection API starting up")
    logger.info("=" * 60)

    # Ensure required directories exist
    UPLOADS_DIR.mkdir(exist_ok=True)
    (UPLOADS_DIR / "results").mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)

    # Pre-load the detection pipeline (loads YOLOv8 weights)
    pipeline = get_pipeline()
    logger.info(f"Model loaded: {pipeline.model_info}")

    yield  # ← server is running

    logger.info("API shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SENORITA — SSS Marine Debris Detection & Geotagging API",
    description=(
        "SENORITA: Edge-AI system for automated detection, classification, "
        "and sub-meter hydrographic geotagging of marine debris in side-scan sonar imagery.\n\n"
        "Detectable classes: ghost nets, shipwrecks, pipes, cylinders, "
        "debris clusters, cables, and unknown anomalies."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Static Files ─────────────────────────────────────────────────────────────

# Serve /static/* from the static/ folder
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve /uploads/* so the frontend can display uploaded/result images
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOADS_DIR)),
    name="uploads",
)


# ─── API Routes ───────────────────────────────────────────────────────────────

app.include_router(router)


# ─── Live WebSocket Manager ───────────────────────────────────────────────────

class LiveConnectionManager:
    """Manages dashboard WebSocket clients for live updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard WS connected ({len(self.active_connections)} total)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Dashboard WS disconnected ({len(self.active_connections)} remaining)")

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)


ws_manager = LiveConnectionManager()


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection telemetry and heartbeats.
    """
    await ws_manager.connect(websocket)
    try:
        pipeline = get_pipeline()
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": "MAIN-SURVEY-01",
            "stats": {
                "model": pipeline.model_info.get("model_name", "yolov8_sonar"),
                "total_pings": 0,
            }
        }))
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "stats": {"model": "yolov8_sonar"}
                }))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        ws_manager.disconnect(websocket)


# ─── Root — Serve Frontend ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_frontend(request: Request):
    """
    Serve the main web UI.
    Falls back to a simple status page if no templates/index.html exists.
    """
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    # Inline fallback page when templates haven't been built yet
    return HTMLResponse(content=_fallback_html())


def _fallback_html() -> str:
    """Minimal HTML status page shown before the full UI is built."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SENORITA — Marine Debris Detection API</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 48px;
    max-width: 560px;
    text-align: center;
    box-shadow: 0 0 60px rgba(30,90,160,0.3);
  }
  .icon { font-size: 3.5rem; margin-bottom: 16px; }
  h1 { font-size: 1.6rem; color: #38bdf8; margin-bottom: 8px; }
  p { color: #94a3b8; line-height: 1.6; margin-bottom: 24px; }
  .badge {
    display: inline-block;
    background: #0f3460;
    color: #38bdf8;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.8rem;
    margin: 4px;
  }
  .links { margin-top: 32px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
  a.btn {
    background: #1e3a5f;
    color: #38bdf8;
    padding: 10px 22px;
    border-radius: 8px;
    text-decoration: none;
    font-size: 0.9rem;
    transition: background 0.2s;
  }
  a.btn:hover { background: #2563eb; color: #fff; }
  .status { margin-top: 16px; color: #22c55e; font-size: 0.85rem; }
</style>
</head>
<body>
  <div class="card">
    <div class="icon">🌊</div>
    <h1>SENORITA</h1>
    <p style="color:#38bdf8; font-weight:600; margin-bottom:8px;">SSS Marine Debris Detection & Geotagging System</p>
    <p>
      Side-scan sonar analysis powered by YOLOv8.
      Upload sonar imagery via the REST API or interactive UI.
    </p>
    <div>
      <span class="badge">Ghost Nets</span>
      <span class="badge">Shipwrecks</span>
      <span class="badge">Pipes</span>
      <span class="badge">Cylinders</span>
      <span class="badge">Cables</span>
      <span class="badge">Debris Clusters</span>
    </div>
    <div class="links">
      <a href="/docs" class="btn">📖 Swagger UI</a>
      <a href="/redoc" class="btn">📄 ReDoc</a>
      <a href="/health" class="btn">💚 Health</a>
      <a href="/api/classes" class="btn">🏷️ Classes</a>
    </div>
    <p class="status">✅ API is online</p>
  </div>
</body>
</html>"""


# ─── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ─── Dev Server Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
