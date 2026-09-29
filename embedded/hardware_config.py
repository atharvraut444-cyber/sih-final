"""
Hardware Configuration Profiles
=================================
Platform-specific settings for edge deployment.

Supported platforms:
  - NVIDIA Jetson Orin Nano  (primary target)
  - NVIDIA Jetson Nano
  - Raspberry Pi 5
  - Windows/Linux x86 PC     (development / vessel PC fallback)

Usage:
    from embedded.hardware_config import get_profile, HardwarePlatform
    profile = get_profile(HardwarePlatform.JETSON_ORIN)
"""

import os
import platform
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class HardwarePlatform(str, Enum):
    JETSON_ORIN   = "jetson_orin"
    JETSON_NANO   = "jetson_nano"
    RPI5          = "rpi5"
    RPI4          = "rpi4"
    WINDOWS_PC    = "windows_pc"
    LINUX_PC      = "linux_pc"
    AUTO          = "auto"          # Auto-detect at runtime


@dataclass
class HardwareProfile:
    """Complete hardware configuration for a deployment target."""

    platform: HardwarePlatform
    name: str

    # ── Compute ──────────────────────────────────────────────────────
    use_gpu: bool = False
    use_tensorrt: bool = False      # TensorRT engine (Jetson only)
    use_onnx: bool = True           # ONNX runtime (universal)
    max_inference_threads: int = 2
    inference_device: str = "cpu"   # "cuda", "cpu", "mps"

    # ── Model ────────────────────────────────────────────────────────
    model_input_size: int = 416     # Smaller = faster on edge
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45

    # ── Sonar Interface ──────────────────────────────────────────────
    default_serial_port: str = "/dev/ttyUSB0"
    default_baud_rate: int = 115200
    default_udp_port: int = 4000
    sonar_range_m: float = 75.0
    ping_interval_m: float = 0.1
    pings_per_frame: int = 512      # How many pings = one image frame
    layback_m: float = 0.0          # Towfish layback behind vessel GPS (T4-D)
    cable_length_m: Optional[float] = None
    towfish_depth_m: float = 0.0

    # ── Storage ──────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path("data"))
    db_path: Path = field(default_factory=lambda: Path("data/detections.db"))
    log_dir: Path = field(default_factory=lambda: Path("data/logs"))
    max_storage_gb: float = 16.0   # Auto-prune oldest data when exceeded

    # ── Display / UI ─────────────────────────────────────────────────
    ui_host: str = "0.0.0.0"
    ui_port: int = 8000
    display_resolution: tuple = (1280, 720)
    has_hdmi: bool = True

    # ── GPIO (Jetson / RPi only) ─────────────────────────────────────
    gpio_available: bool = False
    alert_led_pin: Optional[int] = None   # GPIO pin for critical-alert LED
    status_led_pin: Optional[int] = None  # GPIO pin for system-alive LED

    # ── Power / Thermal ──────────────────────────────────────────────
    target_fps: float = 10.0        # Target detection frames per second
    thermal_throttle_temp_c: float = 80.0

    def ensure_dirs(self):
        """Create required data directories."""
        for d in [self.data_dir, self.log_dir,
                  self.data_dir / "images", self.data_dir / "reports"]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ── Pre-built Profiles ────────────────────────────────────────────────────────

PROFILES = {
    HardwarePlatform.JETSON_ORIN: HardwareProfile(
        platform=HardwarePlatform.JETSON_ORIN,
        name="NVIDIA Jetson Orin Nano",
        use_gpu=True,
        use_tensorrt=True,
        use_onnx=True,
        inference_device="cuda",
        max_inference_threads=4,
        model_input_size=640,
        pings_per_frame=512,
        gpio_available=True,
        alert_led_pin=18,
        status_led_pin=24,
        target_fps=15.0,
        default_serial_port="/dev/ttyTHS0",
    ),
    HardwarePlatform.JETSON_NANO: HardwareProfile(
        platform=HardwarePlatform.JETSON_NANO,
        name="NVIDIA Jetson Nano",
        use_gpu=True,
        use_tensorrt=True,
        use_onnx=True,
        inference_device="cuda",
        max_inference_threads=2,
        model_input_size=416,
        pings_per_frame=416,
        gpio_available=True,
        alert_led_pin=18,
        status_led_pin=24,
        target_fps=8.0,
        default_serial_port="/dev/ttyTHS1",
    ),
    HardwarePlatform.RPI5: HardwareProfile(
        platform=HardwarePlatform.RPI5,
        name="Raspberry Pi 5",
        use_gpu=False,
        use_tensorrt=False,
        use_onnx=True,
        inference_device="cpu",
        max_inference_threads=4,
        model_input_size=320,
        pings_per_frame=320,
        gpio_available=True,
        alert_led_pin=17,
        status_led_pin=27,
        target_fps=5.0,
        default_serial_port="/dev/ttyUSB0",
    ),
    HardwarePlatform.RPI4: HardwareProfile(
        platform=HardwarePlatform.RPI4,
        name="Raspberry Pi 4",
        use_gpu=False,
        use_tensorrt=False,
        use_onnx=True,
        inference_device="cpu",
        max_inference_threads=2,
        model_input_size=320,
        pings_per_frame=320,
        gpio_available=True,
        alert_led_pin=17,
        status_led_pin=27,
        target_fps=3.0,
        default_serial_port="/dev/ttyUSB0",
    ),
    HardwarePlatform.WINDOWS_PC: HardwareProfile(
        platform=HardwarePlatform.WINDOWS_PC,
        name="Windows PC (vessel / development)",
        use_gpu=False,
        use_tensorrt=False,
        use_onnx=True,
        inference_device="cpu",
        max_inference_threads=4,
        model_input_size=640,
        pings_per_frame=512,
        gpio_available=False,
        target_fps=10.0,
        default_serial_port="COM3",
        data_dir=Path("data"),
    ),
    HardwarePlatform.LINUX_PC: HardwareProfile(
        platform=HardwarePlatform.LINUX_PC,
        name="Linux PC (vessel / development)",
        use_gpu=False,
        use_tensorrt=False,
        use_onnx=True,
        inference_device="cpu",
        max_inference_threads=4,
        model_input_size=640,
        pings_per_frame=512,
        gpio_available=False,
        target_fps=10.0,
        default_serial_port="/dev/ttyUSB0",
        data_dir=Path("data"),
    ),
}


def auto_detect_platform() -> HardwarePlatform:
    """Auto-detect the running platform from system information."""
    sys_platform = platform.system().lower()

    if sys_platform == "windows":
        return HardwarePlatform.WINDOWS_PC

    # Linux — check for Jetson / RPi markers
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
        if "orin" in model:
            return HardwarePlatform.JETSON_ORIN
        if "jetson nano" in model:
            return HardwarePlatform.JETSON_NANO
        if "raspberry pi 5" in model:
            return HardwarePlatform.RPI5
        if "raspberry pi" in model:
            return HardwarePlatform.RPI4
    except FileNotFoundError:
        pass

    return HardwarePlatform.LINUX_PC


def get_profile(platform_id: HardwarePlatform = HardwarePlatform.AUTO) -> HardwareProfile:
    """
    Get hardware profile for the given platform.
    If AUTO, detects the current platform automatically.
    """
    if platform_id == HardwarePlatform.AUTO:
        platform_id = auto_detect_platform()

    profile = PROFILES.get(platform_id, PROFILES[HardwarePlatform.LINUX_PC])
    profile.ensure_dirs()
    return profile
