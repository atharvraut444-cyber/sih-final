# SENORITA — SSS Marine Debris Detection & Geotagging System
## Setup & Architecture Guide

## Python Interpreter Issue (Windows)

On **Windows**, there are two Python interpreters present:

| Interpreter | Path | Has cv2/numpy/ultralytics? |
|---|---|---|
| `python` / `python3` | `C:\msys64\ucrt64\bin\python.exe` (Python 3.14) | ❌ No |
| `py -3` | Python 3.13 (Windows Launcher) | ✅ Yes |

**All project scripts must use `py -3` on Windows.**

### Quick Fix — Local Shim

A `python.bat` shim is provided in the project root. Add the project root to your
`PATH` (or `cd` into it before running commands) and `python` will delegate to `py -3`:

```powershell
# From project root
.\python tests\benchmark_ocean.py --n 20 --seed 42
# or
run_benchmark.bat --n 20 --seed 42
```

### Installing Dependencies

```powershell
# Windows
py -3 -m pip install -r requirements.txt

# Linux / macOS
python3 -m pip install -r requirements.txt
```

### Running the Benchmark

```powershell
# Windows (explicit)
py -3 tests\benchmark_ocean.py --n 25 --seed 2025

# Windows (via shim — from project root)
python tests\benchmark_ocean.py --n 25 --seed 2025
run_benchmark.bat --n 25 --seed 2025

# Linux / macOS
python3 tests/benchmark_ocean.py --n 25 --seed 2025
```

### Benchmark Output Files

By default the benchmark generates a **timestamped** output file so previous results
are never overwritten:

```
tests/benchmark_results_20260905_172308.json
```

To specify a custom output path (will not overwrite without `--force`):

```powershell
py -3 tests\benchmark_ocean.py --n 25 --out tests\my_run.json
# If the file already exists, add --force to overwrite:
py -3 tests\benchmark_ocean.py --n 25 --out tests\my_run.json --force
```

### Running the API Server

```powershell
# Windows
py -3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Linux / macOS
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### On Linux / Embedded (Jetson, RPi)

The `deploy/install.sh` script uses `python3` by default which resolves correctly
on all supported Linux targets. No shim needed.

---

## Detection Mode

The system runs in one of two modes:

| Mode | When | Output tag |
|---|---|---|
| **heuristic** | No trained model weights found | `"detection_mode": "heuristic"` |
| **model** | `models/yolov8_sonar.pt` present | `"detection_mode": "model"` |

When in heuristic mode, **a warning is logged at startup** and all JSON reports
include `"detection_mode": "heuristic"`. See `TRAINING_GUIDE.md` for how to
train a real model.
