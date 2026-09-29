# SENORITA — SSS Marine Debris Detection & Geotagging System

An end-to-end AI and edge-compute system for detecting, classifying, and geotagging marine debris from Side-Scan Sonar (SSS) acoustic imagery.

---

## 🌊 System Architecture

```
SSS Sonar Feed / File Upload
           │
           ▼
┌──────────────────────────────────────┐
│  Preprocessing Pipeline (CLAHE, NLM) │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  YOLOv8 Sonar Object Detector        │
│  (7-Class Detection + Shadow Bonus)  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Geotagging & Coordinate Mapping     │
│  (Towfish GPS + Slant-Range Engine)  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Interactive Web UI & Real-Time API  │
│  (FastAPI + Leaflet Mapbox Viewer)   │
└──────────────────────────────────────┘
```

---

## 🚀 Key Features

- **7-Class Underwater Debris Detection**: Ghost Net, Shipwreck, Pipe, Cylinder, Debris Cluster, Cable, and Anomaly.
- **Acoustic Preprocessing**: Adaptive CLAHE, bilateral filtering, and Non-Local Means (NLM) with timeout safety guards.
- **Dual Inference Mode**: Real-time YOLOv8 neural network inference with seamless fallback to acoustic shadow heuristic detection.
- **Precision Geotagging**: Computes geographical coordinates (WGS-84) from towfish telemetry, altitude, and sonar slant-range geometry.
- **Interactive Geospatial Dashboard**: Web-based dashboard displaying sonar swaths, detections, confidence scores, and Leaflet map integration.
- **Edge Deployment Ready**: Embedded modules supporting NVIDIA Jetson edge inference and real-time serial/UDP sonar streams.

---

## 📦 Project Structure

```
├── api/                  # FastAPI REST endpoints & background pipeline runner
├── core/                 # Core detection, preprocessing, geotagging, & confidence engine
├── data/sonar_dataset/   # Curated sonar training & validation dataset
├── deploy/               # Systemd services & automated Linux setup scripts
├── embedded/             # Edge compute runtime & sonar hardware reader
├── models/               # Trained neural network weights (YOLOv8 Sonar)
├── static/               # Frontend assets
├── templates/            # Web UI dashboard (index.html)
├── tests/                # Ocean benchmark test suite
├── training/             # Model training, augmentation, and evaluation scripts
├── config.py             # Central project configuration
├── main.py               # Main application entry point
├── edge_main.py          # Standalone edge inference runner
└── requirements.txt      # Python dependencies
```

---

## 🛠️ Quick Start

### 1. Prerequisites & Dependencies

Python 3.10+ is recommended.

```bash
# Clone the repository
git clone https://github.com/ms5014342/sih-project.git
cd sih-project

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
# Start the FastAPI server & web interface
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Run Benchmark Tests

```bash
python tests/benchmark_ocean.py --n 20 --seed 42
```

---

## 🏷️ Detection Classes

| ID | Class Name | Description |
|---|---|---|
| 0 | `ghost_net` | Abandoned fishing gear & nets |
| 1 | `shipwreck` | Hull fragments & wreckage |
| 2 | `pipe` | Subsea pipelines & conduits |
| 3 | `cylinder` | Drums, barrels, & metallic cylinders |
| 4 | `debris_cluster` | Aggregated seafloor debris fields |
| 5 | `cable` | Underwater cables & mooring lines |
| 6 | `anomaly` | Unclassified acoustic anomalies |

---

## 📜 License

MIT License. See project documentation for further details.
