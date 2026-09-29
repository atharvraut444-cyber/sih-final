"""
SENORITA — Global Configuration for Marine Debris Detection System
===================================================================
Central configuration module containing all tunable parameters,
file paths, model settings, and class definitions.
"""

import os
from pathlib import Path

# ─── Project Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure required directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
(UPLOADS_DIR / "results").mkdir(exist_ok=True)

# ─── Model Configuration ────────────────────────────────────────
MODEL_WEIGHTS = MODELS_DIR / "yolov8_sonar.pt"
MODEL_INPUT_SIZE = 416                 # Aligned with training imgsz=416 (T2-D)
MODEL_CONFIDENCE_THRESHOLD = 0.20      # Optimized for challenging low-contrast conditions (mud, high altitude)
MODEL_IOU_THRESHOLD = 0.45
TILE_OVERLAP = 80                       # Wider overlap for 1000px sonar swaths

# ─── Class Definitions ──────────────────────────────────────────
CLASS_NAMES = {
    0: "ghost_net",
    1: "shipwreck",
    2: "pipe",
    3: "cylinder",
    4: "debris_cluster",
    5: "cable",
    6: "anomaly",
}

CLASS_LABELS = {
    0: "Ghost Net / Fishing Gear",
    1: "Shipwreck / Hull Fragment",
    2: "Industrial Pipe / Conduit",
    3: "Barrel / Cylinder / Drum",
    4: "Debris Cluster",
    5: "Underwater Cable / Rope",
    6: "Unknown Anomaly",
}

CLASS_COLORS = {
    0: "#ff4757",   # Red — ghost nets
    1: "#ffa502",   # Orange — shipwrecks
    2: "#1e90ff",   # Blue — pipes
    3: "#2ed573",   # Green — cylinders
    4: "#ff6348",   # Coral — debris clusters
    5: "#a855f7",   # Purple — cables
    6: "#eccc68",   # Yellow — anomalies
}

# ─── Preprocessing Configuration ────────────────────────────────
PREPROCESSING = {
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,
    "nlm_h": 8,                  # Reduced: less over-smoothing at high altitude
    "nlm_template_window": 7,
    "nlm_search_window": 21,
    "clahe_clip_limit": 4.0,     # Stronger: needed for 100kHz long-range low contrast
    "clahe_tile_grid": (16, 8),  # Wider grid for 1000px-wide sonar swaths
    "nadir_width_ratio": 0.04,   # Slightly wider nadir exclusion
}

# NLM denoising safety limits (Issue 2/4 — timeout guard)
NLM_TIMEOUT_SECONDS = 0.5           # Hard timeout: fallback to bilateral if exceeded
NLM_SEARCH_WINDOW_HIGH_TEXTURE = 11  # Reduced search window on high-texture images (std > threshold)
NLM_HIGH_TEXTURE_STD = 15.0         # std threshold above which we reduce nlm_search_window

# ─── Heuristic Detector Configuration ───────────────────────────
# (applies only in demo/heuristic mode — has no effect when a real model is loaded)

# Roughness-adaptive contrast gate (Issue 1 — FP explosion on coral/rock)
ROUGH_BOTTOM_STD_THRESHOLD  = 18.0  # Image std above which background is classified as rough
ROUGH_BOTTOM_CONTRAST_MIN   = 1.2   # Contrast multiplier required in rough-bottom images
NORMAL_CONTRAST_MIN         = 0.6   # Contrast multiplier for smooth/moderate bottoms

# Compactness gate: reject irregular texture clusters
# compactness = 4π·area / perimeter² ; circle=1, irregular→0
COMPACTNESS_MIN = 0.08              # Below this: blob is too irregular to be a man-made object

# Detection cap per image (prevents hundreds of FPs on coral/rock)
DETECTOR_MAX_DETECTIONS_PER_TILE = 8

# 100kHz frequency-aware relaxation (Issue 3)
FREQ_100KHZ_AREA_SCALE   = 0.70    # min_area multiplied by this factor for 100kHz (objects appear smaller)
FREQ_100KHZ_CONTRAST_MIN = 0.40    # Relaxed contrast minimum for low-frequency long-range sonar

# Shadow bonus multiplier applied to ensemble confidence for confirmed-shadow detections (Issue 5)
SHADOW_BONUS_THRESHOLD   = 0.35    # shadow_score must exceed this to earn the bonus (calibrated for real sonar shadows)
SHADOW_BONUS_MULTIPLIER  = 1.20    # Multiplier applied to final ensemble score

# Calibrated confidence weights: prioritizes validated YOLO model while integrating acoustic physics
CONFIDENCE_WEIGHTS = {
    "yolo_conf":        0.42,   # Primary learned acoustic detection model
    "shadow_score":     0.28,   # Acoustic shadow = physical elevation cue
    "morphology_score": 0.18,   # Contour geometry & aspect ratio
    "texture_score":    0.12,   # Texture entropy vs background
}

# Class-adaptive weights: dynamically adjusts physics weighting based on object acoustic geometry
CLASS_CONFIDENCE_WEIGHTS = {
    "pipe": {
        "yolo_conf": 0.40,
        "shadow_score": 0.26,
        "morphology_score": 0.24,
        "texture_score": 0.10,
    },
    "cylinder": {
        "yolo_conf": 0.40,
        "shadow_score": 0.26,
        "morphology_score": 0.24,
        "texture_score": 0.10,
    },
    "shipwreck": {
        "yolo_conf": 0.42,
        "shadow_score": 0.28,
        "morphology_score": 0.18,
        "texture_score": 0.12,
    },
    "cable": {
        "yolo_conf": 0.44,
        "shadow_score": 0.12,
        "morphology_score": 0.32,
        "texture_score": 0.12,
    },
    "ghost_net": {
        "yolo_conf": 0.42,
        "shadow_score": 0.20,
        "morphology_score": 0.16,
        "texture_score": 0.22,
    },
    "debris_cluster": {
        "yolo_conf": 0.42,
        "shadow_score": 0.22,
        "morphology_score": 0.16,
        "texture_score": 0.20,
    },
    "anomaly": {
        "yolo_conf": 0.42,
        "shadow_score": 0.25,
        "morphology_score": 0.18,
        "texture_score": 0.15,
    },
}

SEVERITY_THRESHOLDS = {
    "critical": 80,   # High-certainty confirmed debris
    "moderate": 60,   # Well-supported detection
    "low":      28,   # Catch low-confidence anomalies (e.g. 100kHz long-range targets) for operator review
}

# ─── Geotagging — Frequency-keyed Sonar Range (Issue 10) ────────
# One-side swath range in metres, from real SSS frequency physics.
# These replace the old single constant DEFAULT_SONAR_RANGE_M=75 which was
# only correct for 300kHz and caused 2x–2.5x position errors for other freqs.
FREQUENCY_SONAR_RANGE_M = {
    "100kHz": 150.0,   # Wide range, lowest resolution
    "300kHz":  75.0,   # Standard survey frequency
    "600kHz":  50.0,
    "900kHz":  30.0,   # Short range, highest resolution
}
DEFAULT_SONAR_RANGE_M = 75.0       # Fallback when frequency is unknown
DEFAULT_PING_INTERVAL_M = 0.1      # Default along-track resolution
DEFAULT_SPEED_KNOTS = 3.0          # Default vessel speed

# ─── API Configuration ──────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 200
ALLOWED_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"}
