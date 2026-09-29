"""
Sonar Image Preprocessing & Noise Filtering Module
====================================================
Handles the unique challenges of side-scan sonar imagery:
- Multiplicative speckle noise reduction
- CLAHE contrast enhancement
- Nadir strip correction
- Acoustic shadow analysis
- Resolution normalization for consistent model input

Pipeline: Raw Sonar → Normalize → Despeckle → Enhance → Tile → Output
"""

import cv2
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PREPROCESSING, MODEL_INPUT_SIZE, TILE_OVERLAP,
    NLM_TIMEOUT_SECONDS, NLM_SEARCH_WINDOW_HIGH_TEXTURE, NLM_HIGH_TEXTURE_STD,
)


# ─── Module-level NLM worker (must be picklable for ProcessPoolExecutor) ─────
# On Windows the 'spawn' start method requires submitted callables to be defined
# at module level so they can be re-imported in the child process.  A lambda or
# nested function would raise a PicklingError.

def _nlm_worker(
    image: np.ndarray, h: float, template_window: int, search_window: int
) -> np.ndarray:
    """Run cv2.fastNlMeansDenoising in a subprocess (called by ProcessPoolExecutor)."""
    import cv2  # Re-import in child process  # noqa: PLC0415
    return cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


@dataclass
class PreprocessedResult:
    """Container for preprocessed sonar image data."""
    original: np.ndarray
    processed: np.ndarray
    tiles: List[np.ndarray] = field(default_factory=list)
    tile_positions: List[Tuple[int, int]] = field(default_factory=list)
    nadir_mask: Optional[np.ndarray] = None
    shadow_map: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


class SonarPreprocessor:
    """
    End-to-end preprocessing pipeline for side-scan sonar imagery.
    
    Converts raw acoustic images into clean, contrast-enhanced tiles
    ready for YOLOv8 inference.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or PREPROCESSING
        self.tile_size = MODEL_INPUT_SIZE
        self.tile_overlap = TILE_OVERLAP
        # Persistent process pool for NLM denoising — spawned once per preprocessor
        # instance and reused for every frame.  Avoids the ~200 ms Windows
        # process-spawn overhead that would otherwise be paid on every NLM call.
        self._nlm_pool: Optional[ProcessPoolExecutor] = None

    def _get_nlm_pool(self) -> ProcessPoolExecutor:
        """Return (and lazily create) the persistent NLM worker pool."""
        if self._nlm_pool is None:
            self._nlm_pool = ProcessPoolExecutor(max_workers=1)
        return self._nlm_pool

    def warm_start(self):
        """Pre-warm the NLM ProcessPoolExecutor to eliminate first-frame latency penalty (T3-A)."""
        pool = self._get_nlm_pool()
        dummy = np.zeros((16, 16), dtype=np.uint8)
        try:
            future = pool.submit(_nlm_worker, dummy, 3.0, 7, 21)
            future.result(timeout=5.0)
            logger.info("SonarPreprocessor NLM pool pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"SonarPreprocessor pool warm start failed: {e}")

    def __del__(self):
        """Shut down the NLM worker pool on garbage-collection."""
        pool = getattr(self, "_nlm_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def process(self, image_path: str) -> PreprocessedResult:
        """
        Run the full preprocessing pipeline on a sonar image.

        Args:
            image_path: Path to the raw sonar image file.

        Returns:
            PreprocessedResult with processed image, tiles, and metadata.
        """
        # Step 1: Load and normalize
        raw = self._load_image(image_path)
        original = raw.copy()

        # Step 2: Detect and mask nadir strip
        nadir_mask = self._detect_nadir(raw)

        # Step 3: Speckle noise reduction (multi-stage)
        despeckled = self._reduce_speckle(raw)

        # Step 4: CLAHE contrast enhancement
        enhanced = self._apply_clahe(despeckled)

        # Step 5: Acoustic shadow analysis
        shadow_map = self._analyze_shadows(enhanced, nadir_mask)

        # Step 6: Final normalization
        normalized = self._normalize_intensity(enhanced)

        # Step 7: Generate tiles for inference
        tiles, positions = self._generate_tiles(normalized)

        return PreprocessedResult(
            original=original,
            processed=normalized,
            tiles=tiles,
            tile_positions=positions,
            nadir_mask=nadir_mask,
            shadow_map=shadow_map,
            metadata={
                "original_shape": raw.shape,
                "processed_shape": normalized.shape,
                "num_tiles": len(tiles),
                "tile_size": self.tile_size,
            },
        )

    def process_array(self, image: np.ndarray) -> PreprocessedResult:
        """
        Run the preprocessing pipeline on a numpy array directly.

        Args:
            image: Raw sonar image as numpy array.

        Returns:
            PreprocessedResult with processed data.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raw = image.astype(np.float64)
        original = image.copy()

        nadir_mask = self._detect_nadir(raw)
        despeckled = self._reduce_speckle(raw)
        enhanced = self._apply_clahe(despeckled)
        shadow_map = self._analyze_shadows(enhanced, nadir_mask)
        normalized = self._normalize_intensity(enhanced)
        tiles, positions = self._generate_tiles(normalized)

        return PreprocessedResult(
            original=original,
            processed=normalized,
            tiles=tiles,
            tile_positions=positions,
            nadir_mask=nadir_mask,
            shadow_map=shadow_map,
            metadata={
                "original_shape": image.shape,
                "processed_shape": normalized.shape,
                "num_tiles": len(tiles),
                "tile_size": self.tile_size,
            },
        )

    # ─── Step 1: Load & Normalize ────────────────────────────────

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load sonar image and convert to grayscale float64."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Sonar image not found: {image_path}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        return image.astype(np.float64)

    # ─── Step 2: Nadir Detection ─────────────────────────────────

    def _detect_nadir(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the nadir strip — the dark central vertical band
        directly below the sonar towfish where no useful data exists.

        Returns a binary mask where 1 = nadir region.
        """
        h, w = image.shape[:2]
        nadir_width = int(w * self.config["nadir_width_ratio"])

        mask = np.zeros((h, w), dtype=np.uint8)

        # Nadir is typically at the center of a side-scan sonar image
        center = w // 2
        half_nadir = nadir_width // 2
        mask[:, max(0, center - half_nadir):min(w, center + half_nadir)] = 1

        # Refine: check if the center column is actually dark
        center_strip = image[:, max(0, center - half_nadir):min(w, center + half_nadir)]
        mean_intensity = np.mean(center_strip)
        overall_mean = np.mean(image)

        # Only mark as nadir if significantly darker than surroundings
        if mean_intensity > overall_mean * 0.5:
            # Not a typical nadir — might be a port/starboard only image
            mask[:] = 0

        return mask

    # ─── Step 3: Speckle Noise Reduction ─────────────────────────

    def _reduce_speckle(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-stage speckle noise reduction pipeline.

        Strategy:
        1. Log-domain transform (converts multiplicative → additive noise)
        2. Bilateral filtering (edge-preserving smoothing)
        3. Non-Local Means denoising (patch-based) — with hard timeout + fallback
        4. Inverse log transform

        The NLM step can be extremely slow on high-backscatter (coral/rock) images.
        If it exceeds NLM_TIMEOUT_SECONDS, we skip it and use the bilateral result
        directly, logging a warning. This bounds worst-case latency per frame.
        """
        # Stage 1: Log-domain transform
        # Speckle noise is multiplicative: observed = clean * noise
        # log(observed) = log(clean) + log(noise) → additive
        log_image = np.log1p(image)  # log(1 + x) for numerical stability

        # Normalize to 0-255 for OpenCV filters
        log_norm = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Stage 2: Bilateral filtering — preserves edges while smoothing
        bilateral = cv2.bilateralFilter(
            log_norm,
            d=self.config["bilateral_d"],
            sigmaColor=self.config["bilateral_sigma_color"],
            sigmaSpace=self.config["bilateral_sigma_space"],
        )

        # Stage 3: Non-Local Means Denoising — with adaptive search window + timeout
        # High-texture images (coral, rock) slow NLM dramatically; use a smaller
        # search window when the image std indicates rough backscatter.
        img_std = float(np.std(log_norm))
        if img_std > NLM_HIGH_TEXTURE_STD:
            search_window = NLM_SEARCH_WINDOW_HIGH_TEXTURE  # Faster on rough images
        else:
            search_window = self.config["nlm_search_window"]

        nlm_result = self._run_nlm_with_timeout(
            bilateral,
            h=self.config["nlm_h"],
            template_window=self.config["nlm_template_window"],
            search_window=search_window,
            fallback=bilateral,
        )

        # Stage 4: Inverse log transform back to spatial domain
        nlm_float = nlm_result.astype(np.float32) / 255.0
        # Scale back to original log range
        log_range = np.max(log_image) - np.min(log_image)
        nlm_scaled = nlm_float * log_range + np.min(log_image)
        despeckled = np.expm1(nlm_scaled)  # exp(x) - 1, inverse of log1p

        # Clip to valid range and cast to uint8
        return np.clip(despeckled, 0, 255).astype(np.uint8)

    def _run_nlm_with_timeout(
        self,
        image: np.ndarray,
        h: float,
        template_window: int,
        search_window: int,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """
        Run cv2.fastNlMeansDenoising with a hard wall-clock timeout.

        Uses ProcessPoolExecutor (not ThreadPoolExecutor) so that the
        timeout actually terminates the work: cv2.fastNlMeansDenoising holds
        the GIL inside a C extension, which means a timed-out *thread* keeps
        running and blocks every subsequent NLM call — producing 10-14 s hangs
        on high-backscatter (coral/rock) images even after the timeout fires.
        A separate *process* is killed by the OS on pool shutdown, so the
        fallback path genuinely bounds worst-case latency.

        The worker must be a module-level function (not a lambda or nested
        function) so it can be pickled by the 'spawn' start method on Windows.
        """
        try:
            pool = self._get_nlm_pool()
            future = pool.submit(_nlm_worker, image, h, template_window, search_window)
            try:
                return future.result(timeout=NLM_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                logger.warning(
                    "NLM denoising timed out (>%.1fs) — using bilateral fallback. "
                    "Image std=%.1f, search_window=%d.",
                    NLM_TIMEOUT_SECONDS, float(np.std(image)), search_window,
                )
                future.cancel()
                # Recycle the pool so the lingering worker process is replaced.
                try:
                    self._nlm_pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._nlm_pool = None
                return fallback
            except Exception as exc:
                logger.warning(
                    "NLM denoising error (%s) — using bilateral fallback.", exc
                )
                return fallback
        except Exception as exc:
            # ProcessPoolExecutor itself failed to start (e.g., in frozen
            # environments or unusual runtime constraints) — fall back silently.
            logger.warning(
                "ProcessPoolExecutor unavailable (%s) — using bilateral fallback.", exc
            )
            return fallback

    # ─── Step 4: CLAHE Contrast Enhancement ──────────────────────

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        CLAHE prevents over-amplification of noise in homogeneous dark
        regions while boosting contrast of debris against the seafloor.
        """
        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=self.config["clahe_tile_grid"],
        )
        enhanced = clahe.apply(image_uint8)

        return enhanced.astype(np.float64)

    # ─── Step 5: Shadow Analysis ─────────────────────────────────

    def _analyze_shadows(
        self, image: np.ndarray, nadir_mask: np.ndarray
    ) -> np.ndarray:
        """
        Analyze acoustic shadows in the sonar image.
        
        Real objects on the seafloor cast acoustic shadows (dark regions
        behind them relative to the sonar). This shadow map can be used
        as a supplementary cue for detection validation.
        
        Returns a shadow probability map (0.0 - 1.0).
        """
        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold to find very dark regions (potential shadows)
        # Shadows are typically < 20% of mean intensity
        mean_val = np.mean(image_uint8[nadir_mask == 0]) if np.any(nadir_mask == 0) else np.mean(image_uint8)
        shadow_threshold = max(10, int(mean_val * 0.2))

        _, shadow_binary = cv2.threshold(
            image_uint8, shadow_threshold, 255, cv2.THRESH_BINARY_INV
        )

        # Remove nadir from shadow map (it's always dark, not a real shadow)
        shadow_binary[nadir_mask == 1] = 0

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_clean = cv2.morphologyEx(shadow_binary, cv2.MORPH_OPEN, kernel)
        shadow_clean = cv2.morphologyEx(shadow_clean, cv2.MORPH_CLOSE, kernel)

        # Create probability map using distance transform
        shadow_dist = cv2.distanceTransform(shadow_clean, cv2.DIST_L2, 5)
        shadow_prob = cv2.normalize(shadow_dist, None, 0, 1.0, cv2.NORM_MINMAX)

        return shadow_prob.astype(np.float32)

    # ─── Step 6: Intensity Normalization ─────────────────────────

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize to 0-255 uint8 for model input."""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    # ─── Step 7: Tiling ──────────────────────────────────────────

    def _generate_tiles(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split the sonar image into overlapping tiles for YOLOv8 inference.
        
        For images smaller than tile_size, the image is padded and returned
        as a single tile. For larger images, overlapping tiles are generated
        so detections at tile boundaries can be merged via NMS.
        
        Returns:
            tiles: List of tile images (each tile_size × tile_size).
            positions: List of (x, y) top-left corner positions of each tile.
        """
        h, w = image.shape[:2]
        tiles = []
        positions = []

        # If image fits in one tile, pad and return
        if h <= self.tile_size and w <= self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
            padded[:h, :w] = image
            tiles.append(padded)
            positions.append((0, 0))
            return tiles, positions

        # Calculate step size (tile_size - overlap)
        step = self.tile_size - self.tile_overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Calculate tile boundaries
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                x_start = max(0, x_end - self.tile_size)
                y_start = max(0, y_end - self.tile_size)

                tile = image[y_start:y_end, x_start:x_end]

                # Pad if necessary (edge tiles)
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded = np.zeros(
                        (self.tile_size, self.tile_size), dtype=np.uint8
                    )
                    padded[: tile.shape[0], : tile.shape[1]] = tile
                    tile = padded

                tiles.append(tile)
                positions.append((x_start, y_start))

        return tiles, positions

    # ─── Utility Methods ─────────────────────────────────────────

    def quick_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Quick enhancement for preview/display purposes.
        Applies only CLAHE without full despeckling pipeline.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_float = image.astype(np.float64)
        return self._normalize_intensity(self._apply_clahe(image_float))

    def get_preprocessing_comparison(
        self, image_path: str
    ) -> Dict[str, np.ndarray]:
        """
        Generate a comparison of each preprocessing stage for visualization.
        Useful for debugging and demonstrating the pipeline.
        """
        raw = self._load_image(image_path)

        stages = {
            "1_raw": cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            "2_despeckled": self._normalize_intensity(self._reduce_speckle(raw)),
            "3_clahe_enhanced": self._normalize_intensity(
                self._apply_clahe(self._reduce_speckle(raw))
            ),
            "4_shadow_map": (
                self._analyze_shadows(
                    self._apply_clahe(self._reduce_speckle(raw)),
                    self._detect_nadir(raw),
                )
                * 255
            ).astype(np.uint8),
        }

        return stages
