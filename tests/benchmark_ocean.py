"""
Ocean SSS Benchmark Test Suite
================================
Generates randomised Side-Scan Sonar environments based on real ocean
acoustic frequency benchmarks and tests the full detection pipeline.

Ocean parameters modelled:
  - SSS frequencies: 100 kHz, 300 kHz, 600 kHz, 900 kHz
  - Sea-state noise: Beaufort 0-6
  - Water column effects: absorption, scattering, reverb
  - Bottom types: sand, mud, rock, gravel (different backscatter)
  - Towfish altitude: 2-15m above seafloor
  - Survey speed: 2-5 knots

Run with:
    python tests/benchmark_ocean.py

Outputs a JSON report to tests/benchmark_results.json
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.preprocessor import SonarPreprocessor
from core.detector import SonarDetector
from core.confidence import ConfidenceScorer
from core.geotagging import GeotaggingEngine
from embedded.database import DetectionDatabase, DetectionRecord
from config import FREQUENCY_SONAR_RANGE_M, DEFAULT_SONAR_RANGE_M

# ─── Ocean Physical Parameters ────────────────────────────────────────────────

SONAR_FREQUENCIES = {
    "100kHz":  {"absorption_dB_per_km": 36,   "beam_width_deg": 1.5, "range_m": 150},
    "300kHz":  {"absorption_dB_per_km": 68,   "beam_width_deg": 0.6, "range_m":  75},
    "600kHz":  {"absorption_dB_per_km": 180,  "beam_width_deg": 0.3, "range_m":  50},
    "900kHz":  {"absorption_dB_per_km": 330,  "beam_width_deg": 0.2, "range_m":  30},
}

BOTTOM_TYPES = {
    "sand":    {"backscatter_dB": -18, "texture": "smooth",   "noise_std": 8},
    "mud":     {"backscatter_dB": -28, "texture": "smooth",   "noise_std": 5},
    "rock":    {"backscatter_dB": -8,  "texture": "rough",    "noise_std": 20},
    "gravel":  {"backscatter_dB": -14, "texture": "moderate", "noise_std": 14},
    "coral":   {"backscatter_dB": -10, "texture": "rough",    "noise_std": 18},
}

SEA_STATE = {
    0: {"noise_dB": 25, "name": "Calm"},
    1: {"noise_dB": 30, "name": "Light Ripples"},
    2: {"noise_dB": 40, "name": "Gentle Waves"},
    3: {"noise_dB": 50, "name": "Slight Seas"},
    4: {"noise_dB": 60, "name": "Moderate Seas"},
    5: {"noise_dB": 65, "name": "Rough Seas"},
    6: {"noise_dB": 70, "name": "Very Rough"},
}

DEBRIS_PROFILES = {
    "ghost_net":      {"rcs_dB": -12, "shadow_ratio": 1.5, "size_m": (5, 20)},
    "shipwreck":      {"rcs_dB":  5,  "shadow_ratio": 3.0, "size_m": (20, 80)},
    "pipe":           {"rcs_dB": -5,  "shadow_ratio": 2.0, "size_m": (3, 30)},
    "cylinder":       {"rcs_dB":  0,  "shadow_ratio": 2.5, "size_m": (1, 5)},
    "debris_cluster": {"rcs_dB": -8,  "shadow_ratio": 1.2, "size_m": (3, 15)},
    "cable":          {"rcs_dB": -18, "shadow_ratio": 1.1, "size_m": (10, 100)},
    "anomaly":        {"rcs_dB": -15, "shadow_ratio": 1.0, "size_m": (0.5, 3)},
}

# ─── Environment Generator ────────────────────────────────────────────────────

@dataclass
class OceanEnvironment:
    env_id: str
    frequency: str
    bottom_type: str
    sea_state: int
    towfish_altitude_m: float
    survey_speed_knots: float
    water_depth_m: float
    latitude: float
    longitude: float
    heading_deg: float
    n_debris_objects: int
    debris_types: List[str]
    image_width: int = 1000
    image_height: int = 512
    description: str = ""

    def summary(self) -> str:
        freq_info = SONAR_FREQUENCIES[self.frequency]
        sea = SEA_STATE[self.sea_state]
        return (
            f"[{self.env_id}] {self.frequency} | {self.bottom_type} bottom | "
            f"Sea state {self.sea_state} ({sea['name']}) | "
            f"Alt {self.towfish_altitude_m:.1f}m | "
            f"{self.n_debris_objects} debris items"
        )


def random_environment(rng: np.random.Generator) -> OceanEnvironment:
    """Generate a fully randomised ocean environment."""
    freq = rng.choice(list(SONAR_FREQUENCIES.keys()))
    bottom = rng.choice(list(BOTTOM_TYPES.keys()))
    sea = int(rng.integers(0, 7))  # Covers Beaufort 0-6 inclusive (upper bound exclusive)
    freq_info = SONAR_FREQUENCIES[freq]

    n_debris = int(rng.integers(1, 8))
    debris_types = list(rng.choice(list(DEBRIS_PROFILES.keys()), size=n_debris))

    return OceanEnvironment(
        env_id=uuid.uuid4().hex[:6].upper(),
        frequency=freq,
        bottom_type=bottom,
        sea_state=sea,
        towfish_altitude_m=float(rng.uniform(2.0, 15.0)),
        survey_speed_knots=float(rng.uniform(2.0, 5.0)),
        water_depth_m=float(rng.uniform(10.0, 200.0)),
        latitude=float(rng.uniform(-30.0, 30.0)),    # Tropical ocean band
        longitude=float(rng.uniform(60.0, 120.0)),   # Indian / Pacific Ocean
        heading_deg=float(rng.uniform(0.0, 359.9)),
        n_debris_objects=n_debris,
        debris_types=debris_types,
    )


# ─── Acoustic Image Synthesis ─────────────────────────────────────────────────

def synthesize_sonar_image(env: OceanEnvironment, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a physically-motivated SSS image for the given ocean environment.

    Models:
      - Frequency-dependent attenuation (range-dependent intensity falloff)
      - Bottom backscatter level from bottom type
      - Speckle noise calibrated to sea state
      - Towfish altitude acoustic shadow zone
      - Debris object reflections with realistic radar cross-section
    """
    W, H = env.image_width, env.image_height
    freq_info = SONAR_FREQUENCIES[env.frequency]
    bottom_info = BOTTOM_TYPES[env.bottom_type]
    sea_info = SEA_STATE[env.sea_state]

    img = np.zeros((H, W), dtype=np.float32)
    half = W // 2

    # ── 1. Bottom backscatter with range-dependent attenuation ──────
    # dB backscatter → linear intensity (0-255 scale)
    base_intensity = 255 * 10 ** (bottom_info["backscatter_dB"] / 20.0 + 1.0)
    base_intensity = float(np.clip(base_intensity, 20, 180))

    # Range-dependent falloff: intensity drops with range (spreading + absorption)
    # x=0 is nadir, x=half is max range
    x_coords = np.arange(half, dtype=np.float32)
    range_m = (x_coords / half) * freq_info["range_m"]
    absorption_factor = np.exp(-2 * freq_info["absorption_dB_per_km"] / 8686 * range_m / 1000)
    spreading_factor = np.maximum(range_m, 1) ** (-1.5)
    spreading_factor = spreading_factor / spreading_factor.max()

    backscatter_profile = base_intensity * absorption_factor * spreading_factor
    backscatter_profile = np.clip(backscatter_profile, 5, 220)

    # Apply to image (left=port, right=stbd, both mirrored)
    img[:, :half] = backscatter_profile[::-1][np.newaxis, :]  # Port (mirror)
    img[:, half:] = backscatter_profile[np.newaxis, :]        # Starboard

    # ── 2. Bottom texture noise ──────────────────────────────────────
    texture_std = bottom_info["noise_std"]
    texture = rng.normal(0, texture_std, (H, W)).astype(np.float32)
    if bottom_info["texture"] == "rough":
        # Add correlated noise for rough bottom
        texture = cv2.GaussianBlur(texture, (5, 5), 2)
        texture *= 2
    elif bottom_info["texture"] == "smooth":
        texture = cv2.GaussianBlur(texture, (9, 9), 3)
    img += texture

    # ── 3. Acoustic nadir zone (directly below towfish) ─────────────
    # Width proportional to altitude (higher = wider nadir gap)
    nadir_half_px = int((env.towfish_altitude_m / freq_info["range_m"]) * half * 1.5)
    nadir_half_px = max(3, min(nadir_half_px, 60))
    nadir_start = half - nadir_half_px
    nadir_end   = half + nadir_half_px
    nadir_intensity = rng.uniform(3, 15)
    img[:, nadir_start:nadir_end] = nadir_intensity

    # ── 4. Sea-state surface reverberation (amplitude modulation) ───
    sea_noise_std = sea_info["noise_dB"] * 0.05
    sea_reverberation = rng.normal(0, sea_noise_std, (H, W)).astype(np.float32)
    # Surface reverberation is stronger at near range
    near_range_weight = np.linspace(1.5, 0.3, half)
    surf_left  = sea_reverberation[:, :half] * near_range_weight[::-1]
    surf_right = sea_reverberation[:, half:] * near_range_weight
    img[:, :half] += surf_left
    img[:, half:] += surf_right

    # ── 5. Debris object reflections ────────────────────────────────
    placed_objects = []
    for debris_type in env.debris_types:
        profile = DEBRIS_PROFILES[debris_type]
        size_range = profile["size_m"]
        obj_size_m = rng.uniform(*size_range)

        # Convert size to pixels (roughly)
        px_per_m = half / freq_info["range_m"]
        obj_w_px = max(4, int(obj_size_m * px_per_m))
        obj_h_px = max(4, int(obj_size_m * px_per_m * rng.uniform(0.3, 1.0)))

        # Cap object pixel size to fit within the available space
        max_obj_px = max(4, (nadir_start - obj_w_px // 2 - 4))
        obj_w_px = min(obj_w_px, max_obj_px)
        obj_h_px = min(obj_h_px, max_obj_px)

        # Position: avoid nadir zone
        side = rng.choice(["port", "stbd"])
        port_lo = obj_w_px // 2 + 2
        port_hi = nadir_start - obj_w_px // 2 - 2
        stbd_lo = nadir_end + obj_w_px // 2 + 2
        stbd_hi = W - obj_w_px // 2 - 2

        # Skip placement if there's no room on either side
        if port_hi <= port_lo and stbd_hi <= stbd_lo:
            continue
        if side == "port" and port_hi <= port_lo:
            side = "stbd"
        if side == "stbd" and stbd_hi <= stbd_lo:
            side = "port"

        if side == "port":
            cx = int(rng.integers(port_lo, port_hi))
        else:
            cx = int(rng.integers(stbd_lo, stbd_hi))
        cy = int(rng.integers(obj_h_px//2 + 2, H - obj_h_px//2 - 2))

        x1 = max(0, cx - obj_w_px//2)
        y1 = max(0, cy - obj_h_px//2)
        x2 = min(W-1, cx + obj_w_px//2)
        y2 = min(H-1, cy + obj_h_px//2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Object intensity (RCS-based, frequency-compensated)
        rcs_dB = profile["rcs_dB"]
        obj_intensity = np.clip(180 + rcs_dB * 2 + rng.normal(0, 10), 120, 250)

        # Draw object
        if debris_type == "cable":
            angle = rng.uniform(0, math.pi)
            pt1 = (int(cx - obj_w_px/2 * math.cos(angle)), int(cy - obj_w_px/2 * math.sin(angle)))
            pt2 = (int(cx + obj_w_px/2 * math.cos(angle)), int(cy + obj_w_px/2 * math.sin(angle)))
            # Draw directly onto img (convert, draw, reassign) — single draw, no dead copy
            tmp = img.astype(np.uint8)
            cv2.line(tmp, pt1, pt2, int(obj_intensity), max(1, obj_h_px))
            img = tmp.astype(np.float32)
        elif debris_type == "cylinder":
            cv2.circle(img, (cx, cy), obj_w_px//2, obj_intensity, -1)  # type: ignore[call-arg]
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), float(obj_intensity), -1)

        # Gaussian smear edges (acoustic diffraction)
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (5, 5), 1.5)

        # Acoustic shadow (opposite from nadir)
        shadow_len_px = int(profile["shadow_ratio"] * obj_h_px)
        shadow_y2 = min(H-1, y2 + shadow_len_px)
        if shadow_y2 > y2 and x2 > x1:
            shadow_intensity = np.clip(img[y2:shadow_y2, x1:x2] * 0.15, 2, 30)
            img[y2:shadow_y2, x1:x2] = shadow_intensity

        placed_objects.append({
            "type": debris_type,
            "cx": cx, "cy": cy,
            "size_m": float(obj_size_m),
            "bbox": [x1, y1, x2, y2],
            "side": side,
        })

    # ── 6. Final clipping and conversion ────────────────────────────
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ─── Benchmark Result ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    env_id: str
    frequency: str
    bottom_type: str
    sea_state: int
    sea_state_name: str
    towfish_altitude_m: float
    n_ground_truth: int
    n_detected: int
    detection_rate_pct: float
    false_positive_rate_pct: float
    avg_confidence: float
    preprocessing_ms: float
    detection_ms: float
    scoring_ms: float
    total_ms: float
    detections_by_class: Dict[str, int]
    detections_by_severity: Dict[str, int]
    error: str = ""


# ─── Main Benchmark Runner ────────────────────────────────────────────────────

def run_benchmark(
    n_environments: int = 20,
    seed: int = None,
    output_path: str = "tests/benchmark_results.json",
) -> List[BenchmarkResult]:

    seed = seed or int(time.time())
    rng = np.random.default_rng(seed)
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  SSS Ocean Benchmark Suite")
    print(f"  Seed: {seed} | Environments: {n_environments}")
    print(f"{'='*60}\n")

    # Initialize pipeline
    preprocessor = SonarPreprocessor()
    preprocessor.warm_start()
    detector = SonarDetector()
    scorer = ConfidenceScorer()
    # GeotaggingEngine range is set per-environment from freq config (Issue 10)
    geo = GeotaggingEngine()  # range_m updated inside loop

    results: List[BenchmarkResult] = []
    pass_count = 0

    for i in range(n_environments):
        env = random_environment(rng)
        print(f"[{i+1:02d}/{n_environments}] {env.summary()}")

        try:
            # Synthesize sonar image
            img = synthesize_sonar_image(env, rng)

            # ── Preprocessing ──────────────────────────────────────────
            t0 = time.perf_counter()
            prep = preprocessor.process_array(img)
            t_prep = (time.perf_counter() - t0) * 1000

            # ── Detection ──────────────────────────────────────────────
            t0 = time.perf_counter()
            det_result = detector.detect_tiled(
                tiles=prep.tiles,
                tile_positions=prep.tile_positions,
                full_image_shape=prep.processed.shape[:2],
                frequency_hint=env.frequency,  # enables frequency-aware sensitivity
            )
            t_det = (time.perf_counter() - t0) * 1000

            # ── Confidence Scoring ────────────────────────────────────
            t0 = time.perf_counter()
            scored = scorer.score_detections(
                detections=det_result.detections,
                image=prep.processed,
                shadow_map=prep.shadow_map,
            )
            filtered = scorer.filter_detections(scored)
            t_score = (time.perf_counter() - t0) * 1000

            # ── Geotagging — use frequency-correct sonar range (Issue 10) & slant range (T4-C) ──
            geotagged = geo.geotag_detections(
                scored_detections=filtered,
                image_shape=prep.processed.shape[:2],
                origin_lat=env.latitude,
                origin_lon=env.longitude,
                heading_deg=env.heading_deg,
                towfish_altitude_m=env.towfish_altitude_m,
                frequency=env.frequency,
            )

            t_total = t_prep + t_det + t_score

            # ── Compute metrics ───────────────────────────────────────
            n_gt  = env.n_debris_objects
            n_det = len(geotagged)

            # Simple detection rate (capped at 100%)
            det_rate = min(100.0, (n_det / max(n_gt, 1)) * 100)
            # False positives: detected beyond ground truth count
            fp_rate = max(0.0, ((n_det - n_gt) / max(n_gt, 1)) * 100)

            avg_conf = (
                sum(g.confidence for g in geotagged) / len(geotagged)
                if geotagged else 0.0
            )

            by_class = {}
            by_sev = {}
            for g in geotagged:
                by_class[g.class_name] = by_class.get(g.class_name, 0) + 1
                by_sev[g.severity] = by_sev.get(g.severity, 0) + 1

            res = BenchmarkResult(
                env_id=env.env_id,
                frequency=env.frequency,
                bottom_type=env.bottom_type,
                sea_state=env.sea_state,
                sea_state_name=SEA_STATE[env.sea_state]["name"],
                towfish_altitude_m=env.towfish_altitude_m,
                n_ground_truth=n_gt,
                n_detected=n_det,
                detection_rate_pct=round(det_rate, 1),
                false_positive_rate_pct=round(fp_rate, 1),
                avg_confidence=round(avg_conf, 1),
                preprocessing_ms=round(t_prep, 1),
                detection_ms=round(t_det, 1),
                scoring_ms=round(t_score, 1),
                total_ms=round(t_total, 1),
                detections_by_class=by_class,
                detections_by_severity=by_sev,
            )

            status = "PASS" if n_det > 0 or n_gt == 0 else "MISS"
            if status == "PASS":
                pass_count += 1

            print(
                f"         {status} | GT:{n_gt} Det:{n_det} "
                f"Rate:{det_rate:.0f}% Conf:{avg_conf:.1f}% "
                f"Time:{t_total:.0f}ms"
            )

        except Exception as e:
            import traceback
            res = BenchmarkResult(
                env_id=env.env_id, frequency=env.frequency,
                bottom_type=env.bottom_type, sea_state=env.sea_state,
                sea_state_name=SEA_STATE[env.sea_state]["name"],
                towfish_altitude_m=env.towfish_altitude_m,
                n_ground_truth=env.n_debris_objects, n_detected=0,
                detection_rate_pct=0, false_positive_rate_pct=0,
                avg_confidence=0, preprocessing_ms=0,
                detection_ms=0, scoring_ms=0, total_ms=0,
                detections_by_class={}, detections_by_severity={},
                error=str(e),
            )
            print(f"         ERROR: {e}")
            traceback.print_exc()

        results.append(res)
        import gc
        gc.collect()

    # ── Aggregate Stats ──────────────────────────────────────────────────
    valid = [r for r in results if not r.error]
    total_gt  = sum(r.n_ground_truth for r in valid)
    total_det = sum(r.n_detected for r in valid)
    avg_rate  = sum(r.detection_rate_pct for r in valid) / max(len(valid), 1)
    avg_time  = sum(r.total_ms for r in valid) / max(len(valid), 1)
    avg_conf  = sum(r.avg_confidence for r in valid) / max(len(valid), 1)

    # By frequency
    freq_stats = {}
    for freq in SONAR_FREQUENCIES:
        freq_results = [r for r in valid if r.frequency == freq]
        if freq_results:
            freq_stats[freq] = {
                "count": len(freq_results),
                "avg_detection_rate": round(sum(r.detection_rate_pct for r in freq_results) / len(freq_results), 1),
                "avg_time_ms": round(sum(r.total_ms for r in freq_results) / len(freq_results), 1),
            }

    # By sea state
    sea_stats = {}
    for ss in range(7):
        ss_results = [r for r in valid if r.sea_state == ss]
        if ss_results:
            sea_stats[str(ss)] = {
                "name": SEA_STATE[ss]["name"],
                "count": len(ss_results),
                "avg_detection_rate": round(sum(r.detection_rate_pct for r in ss_results) / len(ss_results), 1),
            }

    summary = {
        "benchmark_config": {
            "seed": seed,
            "n_environments": n_environments,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "overall": {
            "environments_tested": len(results),
            "environments_passed": pass_count,
            "pass_rate_pct": round(pass_count / max(len(results), 1) * 100, 1),
            "total_ground_truth_objects": total_gt,
            "total_detected_objects": total_det,
            "avg_detection_rate_pct": round(avg_rate, 1),
            "avg_processing_time_ms": round(avg_time, 1),
            "avg_confidence_pct": round(avg_conf, 1),
            "target_fps_achievable": round(1000 / max(avg_time, 1), 1),
        },
        "by_frequency": freq_stats,
        "by_sea_state": sea_stats,
        "environments": [asdict(r) for r in results],
    }

    # Save report
    out_path = Path(output_path)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Environments:      {len(results)}")
    print(f"  Pass rate:         {summary['overall']['pass_rate_pct']}%")
    print(f"  Avg detection:     {avg_rate:.1f}%")
    print(f"  Avg confidence:    {avg_conf:.1f}%")
    print(f"  Avg process time:  {avg_time:.0f}ms")
    print(f"  Est. max FPS:      {summary['overall']['target_fps_achievable']:.1f}")
    print(f"\n  By Frequency:")
    for freq, stats in freq_stats.items():
        print(f"    {freq:<8}  rate={stats['avg_detection_rate']}%  time={stats['avg_time_ms']}ms")
    print(f"\n  By Sea State:")
    for ss, stats in sea_stats.items():
        print(f"    SS{ss} {stats['name']:<18}  rate={stats['avg_detection_rate']}%")
    print(f"\n  Report saved: {out_path}")
    print(f"{'='*60}\n")

    # ── Database layer round-trip (Issue 15) ──────────────────────────────────
    # Write all benchmark results through the DetectionDatabase so the DB
    # layer has real test coverage and schema bugs surface here, not in prod.
    db_path = out_path.parent / "benchmark_db_test.sqlite"
    db_errors = []
    try:
        db = DetectionDatabase(db_path=str(db_path))
        session_id = f"bench_{seed}_{time.strftime('%Y%m%d%H%M%S')}"
        db.create_session(
            session_id=session_id,
            interface="benchmark",
            platform="dev",
            notes=f"seed={seed} envs={n_environments}",
        )
        written = 0
        for r in valid:
            rec = DetectionRecord(
                session_id=session_id,
                detection_id=f"{r.env_id}-summary",
                timestamp=time.time(),
                class_name="benchmark_env",
                class_label="Benchmark Environment",
                confidence=r.avg_confidence,
                severity="low",
                latitude=0.0,
                longitude=0.0,
                depth_m=0.0,
                bbox_x1=0, bbox_y1=0, bbox_x2=0, bbox_y2=0,
                est_length_m=0.0,
                est_width_m=0.0,
                yolo_score=0.0,
                shadow_score=0.0,
                morph_score=0.0,
                texture_score=0.0,
                frame_number=0,
            )
            db.log_detection(rec)
            written += 1
        db.end_session(session_id, total_pings=n_environments, total_detections=written)
        read_back = db.get_session_detections(session_id)
        assert len(read_back) == written, (
            f"DB round-trip mismatch: wrote {written}, read back {len(read_back)}"
        )
        print(f"  DB round-trip:     OK ({written} records written & verified)")
    except Exception as db_exc:
        db_errors.append(str(db_exc))
        print(f"  DB round-trip:     FAILED — {db_exc}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="SSS Ocean Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output file:\n"
            "  By default a timestamped file is created so previous runs are\n"
            "  never silently overwritten. Use --force to allow overwriting.\n\n"
            "Example:\n"
            "  py -3 tests/benchmark_ocean.py --n 25 --seed 2025 --force"
        ),
    )
    p.add_argument("--n",     type=int,  default=20,   help="Number of test environments")
    p.add_argument("--seed",  type=int,  default=None,  help="Random seed")
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSON path. Defaults to a timestamped file in tests/ so existing "
            "results are not overwritten. Requires --force if the file already exists."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out file if it already exists (required when --out targets an existing file).",
    )
    args = p.parse_args()

    # Resolve output path — auto-timestamp when not specified
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"tests/benchmark_results_{ts}.json"
    else:
        out_path = args.out

    # Overwrite guard
    if Path(out_path).exists() and not args.force:
        print(
            f"ERROR: output file '{out_path}' already exists.\n"
            "  Add --force to overwrite, or omit --out to get a new timestamped file."
        )
        sys.exit(1)

    run_benchmark(n_environments=args.n, seed=args.seed, output_path=out_path)
