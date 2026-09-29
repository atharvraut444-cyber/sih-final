"""
Sonar-Specific Augmentation Utilities
======================================
Offline augmentation pipeline for side-scan sonar (SSS) imagery.

These augmentations simulate real SSS artefacts and are designed to
complement YOLOv8's built-in online augmentation (mosaic, flips, etc.)
by adding domain-specific transforms that generic vision libraries lack.

Usage:
    from training.augment import SonarAugmentor
    aug = SonarAugmentor(seed=42)
    aug_img, aug_labels = aug(image, labels)   # labels: list of [cls,cx,cy,w,h]

    # Or augment an entire dataset folder:
    python training/augment.py --src data/sonar_dataset --dst data/sonar_augmented --factor 4
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# --- Individual Augmentation Transforms -----------------------------------

def add_speckle_noise(img: np.ndarray, intensity: float = 0.08, rng=None) -> np.ndarray:
    """
    Add multiplicative speckle noise (Rayleigh-distributed) -- the dominant
    noise model for coherent sonar systems.

    Args:
        img:       Grayscale or BGR image (uint8)
        intensity: Noise strength [0..1]. 0.05-0.15 is realistic for SSS.
        rng:       Optional NumPy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = intensity * 255
    speckle = rng.rayleigh(scale=sigma, size=gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + speckle - speckle.mean()
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        noisy = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    return noisy


def intensity_jitter(img: np.ndarray, gain_range=(0.6, 1.5), bias_range=(-25, 25), rng=None) -> np.ndarray:
    """
    Simulate different sonar gain settings and AGC variation.

    Args:
        img:        Input image (uint8, grayscale or BGR)
        gain_range: Multiplicative factor range -- simulates TVG changes.
        bias_range: Additive brightness offset -- simulates backscatter variation.
        rng:        Optional NumPy Generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    gain = rng.uniform(*gain_range)
    bias = rng.uniform(*bias_range)
    result = img.astype(np.float32) * gain + bias
    return np.clip(result, 0, 255).astype(np.uint8)


def nadir_mask(img: np.ndarray, width_ratio: float = 0.04, rng=None) -> np.ndarray:
    """
    Apply and randomize the nadir (zero-range) dark band in the center of the image.

    The nadir is the acoustic blind zone directly below the sonar transducer.
    Varying its width teaches the model to be robust to different nadir sizes.

    Args:
        img:         Input image (uint8)
        width_ratio: Fraction of image width for nadir band (default 4%).
        rng:         Optional NumPy Generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    cx = w // 2
    actual_ratio = rng.uniform(width_ratio * 0.5, width_ratio * 1.5)
    nadir_w = max(4, int(w * actual_ratio))
    x1 = max(0, cx - nadir_w // 2)
    x2 = min(w - 1, cx + nadir_w // 2)
    result = img.copy()
    if img.ndim == 3:
        result[:, x1:x2, :] = rng.integers(3, 18, (h, x2 - x1, img.shape[2]), dtype=np.uint8)
    else:
        result[:, x1:x2] = rng.integers(3, 18, (h, x2 - x1), dtype=np.uint8)
    return result


def horizontal_stripe_noise(img: np.ndarray, n_stripes: int = 3, rng=None) -> np.ndarray:
    """
    Add horizontal line artefacts that mimic sonar ping dropouts or
    sea-surface reverberation bands.

    Args:
        img:      Input image
        n_stripes: Maximum number of stripe artefacts to inject.
        rng:      Optional NumPy Generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    h = img.shape[0]
    result = img.copy()
    count = rng.integers(1, max(2, n_stripes + 1))
    for _ in range(count):
        y = int(rng.integers(0, h))
        stripe_h = int(rng.integers(1, 5))
        intensity = int(rng.integers(0, 40))
        y2 = min(h, y + stripe_h)
        if img.ndim == 3:
            result[y:y2, :, :] = intensity
        else:
            result[y:y2, :] = intensity
    return result


def simulate_acoustic_shadow(img: np.ndarray, rng=None) -> np.ndarray:
    """
    Randomly darken elongated regions to simulate acoustic shadows cast by
    bottom objects. Placed preferentially on the distal (far-range) side.

    This augmentation helps the model learn shadow-brightness pairing even
    in images where the bright highlight may be clipped.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    result = img.copy()
    n = int(rng.integers(1, 4))
    for _ in range(n):
        port = rng.random() > 0.5
        if port:
            x_center = int(rng.uniform(0.1, 0.42) * w)
        else:
            x_center = int(rng.uniform(0.58, 0.9) * w)
        y_center = int(rng.uniform(0.05, 0.95) * h)
        sw = int(rng.uniform(0.03, 0.12) * w)
        sh = int(rng.uniform(0.01, 0.05) * h)
        x1 = max(0, x_center - sw // 2)
        x2 = min(w, x_center + sw // 2)
        y1 = max(0, y_center)
        y2 = min(h, y_center + sh)
        if x2 > x1 and y2 > y1:
            shadow = result[y1:y2, x1:x2].astype(np.int16) - int(rng.uniform(40, 90))
            result[y1:y2, x1:x2] = np.clip(shadow, 3, 255).astype(np.uint8)
    return result


def random_gamma(img: np.ndarray, gamma_range=(0.6, 1.8), rng=None) -> np.ndarray:
    """
    Apply random gamma correction to simulate non-linear sonar response curves.

    Args:
        img:         Input image (uint8)
        gamma_range: Range of gamma values. <1 brightens; >1 darkens.
        rng:         Optional NumPy Generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    gamma = rng.uniform(*gamma_range)
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def random_horizontal_flip(img: np.ndarray, labels: list, rng=None):
    """
    Flip image left-right (port-starboard) and adjust YOLO labels.

    Args:
        img:    Image array (H x W or H x W x C)
        labels: List of [cls, cx, cy, w, h] normalized YOLO labels
        rng:    Optional NumPy Generator.

    Returns:
        (flipped_img, adjusted_labels)
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        labels = [[cls, 1.0 - cx, cy, w, h] for cls, cx, cy, w, h in labels]
    return img, labels


def random_vertical_flip(img: np.ndarray, labels: list, rng=None):
    """
    Flip image top-bottom (along-track direction) and adjust YOLO labels.

    This simulates the sonar towing in the opposite direction.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < 0.3:
        img = cv2.flip(img, 0)
        labels = [[cls, cx, 1.0 - cy, w, h] for cls, cx, cy, w, h in labels]
    return img, labels


# --- Composed Augmentor ---------------------------------------------------

class SonarAugmentor:
    """
    Composed augmentation pipeline for SSS imagery.

    Applies a randomized subset of sonar-domain augmentations.
    All transforms preserve YOLO-format bounding box labels.

    Example::

        aug = SonarAugmentor(seed=42)
        for img, labels in dataset:
            aug_img, aug_labels = aug(img, labels)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        speckle_p: float = 0.7,
        jitter_p: float = 0.8,
        nadir_p: float = 0.6,
        stripe_p: float = 0.3,
        shadow_p: float = 0.4,
        gamma_p: float = 0.5,
        hflip_p: float = 0.5,
        vflip_p: float = 0.3,
    ):
        """
        Args:
            seed:      RNG seed for reproducibility.
            *_p:       Probability of applying each augmentation.
        """
        self.rng = np.random.default_rng(seed)
        self.probs = {
            "speckle": speckle_p,
            "jitter":  jitter_p,
            "nadir":   nadir_p,
            "stripe":  stripe_p,
            "shadow":  shadow_p,
            "gamma":   gamma_p,
            "hflip":   hflip_p,
            "vflip":   vflip_p,
        }

    def __call__(self, img: np.ndarray, labels: list) -> tuple:
        """
        Apply random augmentations to an image and its YOLO labels.

        Args:
            img:    Input image (uint8, grayscale or BGR)
            labels: List of [cls, cx, cy, w, h] normalized YOLO labels.

        Returns:
            (augmented_image, augmented_labels)
        """
        def roll(key: str) -> bool:
            return self.rng.random() < self.probs[key]

        # Spatial augmentations (modify labels)
        img, labels = random_horizontal_flip(img, labels, self.rng)
        img, labels = random_vertical_flip(img, labels, self.rng)

        # Pixel-level augmentations (don't affect labels)
        if roll("jitter"):
            img = intensity_jitter(img, rng=self.rng)
        if roll("gamma"):
            img = random_gamma(img, rng=self.rng)
        if roll("speckle"):
            img = add_speckle_noise(img, rng=self.rng)
        if roll("stripe"):
            img = horizontal_stripe_noise(img, rng=self.rng)
        if roll("shadow"):
            img = simulate_acoustic_shadow(img, rng=self.rng)
        if roll("nadir"):
            img = nadir_mask(img, rng=self.rng)

        return img, labels


# --- Offline Dataset Augmentation CLI ------------------------------------

def augment_dataset(src_dir: Path, dst_dir: Path, factor: int = 3, seed: int = 42):
    """
    Augment an existing YOLO-format dataset by `factor` x copies.

    Creates `dst_dir` with the same YOLO structure (images/ + labels/) and
    writes the original images plus factor-1 augmented variants per image.

    Args:
        src_dir: Source dataset directory (contains images/train, labels/train, etc.)
        dst_dir: Destination for augmented dataset
        factor:  Total copies per original image (1 = original only)
        seed:    RNG seed
    """
    aug = SonarAugmentor(seed=seed)
    for split in ("train", "val"):
        src_img_dir = src_dir / "images" / split
        src_lbl_dir = src_dir / "labels" / split
        if not src_img_dir.exists():
            logger.warning(f"Skipping split '{split}' -- not found: {src_img_dir}")
            continue

        dst_img_dir = dst_dir / "images" / split
        dst_lbl_dir = dst_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(src_img_dir.glob("*.png")) + sorted(src_img_dir.glob("*.jpg"))
        logger.info(f"[{split}] Augmenting {len(img_files)} images x{factor}")

        for img_path in img_files:
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")

            img = cv2.imread(str(img_path))
            labels = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        vals = [float(v) for v in line.strip().split()]
                        if len(vals) == 5:
                            labels.append(vals)

            # Copy original
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

            # Write augmented variants
            for i in range(1, factor):
                aug_img, aug_labels = aug(img.copy(), [list(l) for l in labels])
                stem = f"{img_path.stem}_aug{i:02d}"
                cv2.imwrite(str(dst_img_dir / f"{stem}.png"), aug_img)
                with open(dst_lbl_dir / f"{stem}.txt", "w") as f:
                    for lbl in aug_labels:
                        f.write(" ".join(f"{v:.6f}" for v in lbl) + "\n")

    # Copy dataset.yaml with updated path
    src_yaml = src_dir / "dataset.yaml"
    if src_yaml.exists():
        import yaml
        with open(src_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["path"] = str(dst_dir.resolve())
        with open(dst_dir / "dataset.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        logger.info(f"dataset.yaml written to {dst_dir}")

    logger.info("Augmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline sonar dataset augmentation")
    parser.add_argument("--src", required=True, help="Source YOLO dataset directory")
    parser.add_argument("--dst", required=True, help="Output directory for augmented dataset")
    parser.add_argument("--factor", type=int, default=3,
                        help="Total image copies per original (default 3 = 2 augmented per original)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_dataset(Path(args.src), Path(args.dst), args.factor, args.seed)
