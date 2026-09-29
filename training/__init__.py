"""
Training package for the Marine Debris Sonar Detection System.

Modules:
    train      - YOLOv8 fine-tuning pipeline + synthetic data generator
    augment    - Sonar-specific offline augmentation (speckle, shadow, etc.)
    evaluate   - Post-training metrics, mAP, speed benchmarks
    export_model - ONNX and TensorRT export utilities
"""

from training.train import (
    train,
    generate_synthetic_dataset,
    create_dataset_yaml,
)
from training.augment import SonarAugmentor
from training.evaluate import run_validation, benchmark_speed

__all__ = [
    "train",
    "generate_synthetic_dataset",
    "create_dataset_yaml",
    "SonarAugmentor",
    "run_validation",
    "benchmark_speed",
]
