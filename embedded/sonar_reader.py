"""
Pluggable Sonar Reader Interface
==================================
Provides a unified API to read raw sonar ping data from multiple
SSS hardware sources:

  SerialSonarReader  — RS-232 / RS-422 serial port (EdgeTech 2000, Klein)
  UDPSonarReader     — Ethernet/UDP stream (EdgeTech 4125, Imagenex 881)
  XTFSonarReader     — XTF (eXtended Triton Format) file or live stream
  FileSonarReader    — Replay PNG/TIFF sonar image files (dev/testing)

All readers produce `SonarPing` objects consumed by the RealtimeEngine.
"""

import abc
import socket
import struct
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Iterator, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class SonarPing:
    """
    One horizontal sonar ping — a single scan line.

    In SSS imagery, many pings stacked vertically form a sonar image.
    port_data:      Intensity samples for the port (left) side.
    starboard_data: Intensity samples for the starboard (right) side.
    ping_number:    Sequential ping counter.
    timestamp:      Unix timestamp of the ping.
    latitude:       Optional GPS latitude at ping time.
    longitude:      Optional GPS longitude at ping time.
    heading_deg:    Optional vessel heading in degrees.
    speed_knots:    Optional vessel speed.
    altitude_m:     Optional towfish altitude above seafloor.
    """
    port_data: np.ndarray
    starboard_data: np.ndarray
    ping_number: int = 0
    timestamp: float = field(default_factory=time.time)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading_deg: float = 0.0
    speed_knots: float = 0.0
    altitude_m: float = 0.0

    @property
    def combined(self) -> np.ndarray:
        """Port + starboard side-by-side as a single row (uint8)."""
        port = self._norm(self.port_data)
        stbd = self._norm(self.starboard_data)
        # Mirror port (it scans right-to-left) then join
        return np.concatenate([port[::-1], stbd])

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return arr


# ─── Abstract Base ────────────────────────────────────────────────────────────

class SonarReader(abc.ABC):
    """
    Abstract sonar reader.
    Subclasses implement `_read_loop` which calls `_emit(ping)` for each ping.
    Consumers pull pings from `get_ping()` or iterate via `__iter__`.
    """

    def __init__(self, queue_size: int = 256):
        self._queue: Queue[SonarPing] = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ping_count = 0
        self._callbacks: List[Callable[[SonarPing], None]] = []

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        """Start the reader thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, name=f"{self.__class__.__name__}", daemon=True
        )
        self._thread.start()
        logger.info(f"{self.__class__.__name__} started")

    def stop(self):
        """Stop the reader thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info(f"{self.__class__.__name__} stopped (total pings: {self._ping_count})")

    def get_ping(self, timeout: float = 1.0) -> Optional[SonarPing]:
        """Blocking get — returns next ping or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def __iter__(self) -> Iterator[SonarPing]:
        """Iterate over pings as they arrive (blocks between pings)."""
        while self._running:
            ping = self.get_ping(timeout=0.5)
            if ping is not None:
                yield ping

    def add_callback(self, fn: Callable[[SonarPing], None]):
        """Register a callback called synchronously on each ping."""
        self._callbacks.append(fn)

    @property
    def ping_count(self) -> int:
        return self._ping_count

    # ── Internal ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def _read_loop(self):
        """Override in subclass — runs in a daemon thread."""
        ...

    def _emit(self, ping: SonarPing):
        """Called by subclass for each new ping."""
        ping.ping_number = self._ping_count
        self._ping_count += 1
        if not self._queue.full():
            self._queue.put_nowait(ping)
        else:
            logger.warning("Ping queue full — dropping oldest ping")
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(ping)
        for cb in self._callbacks:
            try:
                cb(ping)
            except Exception as e:
                logger.error(f"Ping callback error: {e}")


# ─── Serial Reader ────────────────────────────────────────────────────────────

class SerialSonarReader(SonarReader):
    """
    RS-232 / RS-422 serial reader for SSS hardware.

    Expects each ping as a binary packet:
      [SYNC 0xFACE][uint16 samples_per_side][float32[] port][float32[] stbd]
      [float64 lat][float64 lon][float32 heading][float32 speed]

    Configurable baud rate — typical SSS units use 115200 or 230400.
    """

    SYNC_WORD = b'\xFA\xCE'

    def __init__(self, port: str, baud_rate: int = 115200,
                 samples_per_side: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.port = port
        self.baud_rate = baud_rate
        self.samples_per_side = samples_per_side

    def _read_loop(self):
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed: pip install pyserial")
            return

        logger.info(f"Opening serial port {self.port} @ {self.baud_rate} baud")
        try:
            ser = serial.Serial(
                self.port,
                baudrate=self.baud_rate,
                timeout=2.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
        except Exception as e:
            logger.error(f"Serial port error: {e}")
            return

        BYTES_PER_SIDE = self.samples_per_side * 4  # float32
        META_BYTES = 8 + 8 + 4 + 4  # lat + lon + heading + speed

        with ser:
            buf = b""
            while self._running:
                buf += ser.read(1024)

                # Hunt for sync word
                idx = buf.find(self.SYNC_WORD)
                if idx < 0:
                    buf = buf[-len(self.SYNC_WORD):]
                    continue
                buf = buf[idx + 2:]  # Consume sync

                # Read samples count
                if len(buf) < 2:
                    buf += ser.read(2 - len(buf))
                n_samples = struct.unpack_from(">H", buf)[0]
                buf = buf[2:]

                # Read ping data
                needed = 2 * n_samples * 4 + META_BYTES
                while len(buf) < needed:
                    buf += ser.read(needed - len(buf))

                port_arr = np.frombuffer(buf[:n_samples * 4], dtype=">f4").astype(np.float32)
                stbd_arr = np.frombuffer(buf[n_samples * 4: 2 * n_samples * 4], dtype=">f4").astype(np.float32)
                meta = buf[2 * n_samples * 4: 2 * n_samples * 4 + META_BYTES]
                lat, lon, hdg, spd = struct.unpack(">ddf f", meta)
                buf = buf[needed:]

                self._emit(SonarPing(
                    port_data=port_arr,
                    starboard_data=stbd_arr,
                    timestamp=time.time(),
                    latitude=lat if lat != 0 else None,
                    longitude=lon if lon != 0 else None,
                    heading_deg=hdg,
                    speed_knots=spd,
                ))


# ─── UDP Reader ───────────────────────────────────────────────────────────────

class UDPSonarReader(SonarReader):
    """
    UDP/Ethernet sonar reader.

    Supports EdgeTech 4125, Imagenex 881, and generic UDP sonar streams.
    Each UDP datagram is one complete ping packet in the same binary format
    as SerialSonarReader, or optionally raw 16-bit intensity array.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 4000,
                 samples_per_side: int = 500, raw_mode: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.samples_per_side = samples_per_side
        self.raw_mode = raw_mode  # If True, packet = raw uint16 array (no header)

    def _read_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        sock.bind((self.host, self.port))
        logger.info(f"UDP sonar listening on {self.host}:{self.port}")

        while self._running:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"UDP recv error: {e}")
                continue

            try:
                if self.raw_mode:
                    # Raw uint16 array: first half=port, second half=stbd
                    arr = np.frombuffer(data, dtype=">u2").astype(np.float32)
                    half = len(arr) // 2
                    self._emit(SonarPing(
                        port_data=arr[:half],
                        starboard_data=arr[half:],
                    ))
                else:
                    # Structured packet with GPS header (same as serial)
                    self._parse_structured_packet(data)
            except Exception as e:
                logger.warning(f"Malformed UDP packet from {addr}: {e}")

        sock.close()

    def _parse_structured_packet(self, data: bytes):
        if len(data) < 6:
            return
        n_samples = struct.unpack_from(">H", data, 0)[0]
        offset = 2
        needed = 2 * n_samples * 4 + 24  # data + lat/lon/hdg/spd
        if len(data) < needed:
            return
        port = np.frombuffer(data[offset: offset + n_samples * 4], dtype=">f4").astype(np.float32)
        stbd = np.frombuffer(data[offset + n_samples * 4: offset + 2 * n_samples * 4], dtype=">f4").astype(np.float32)
        meta_off = offset + 2 * n_samples * 4
        lat, lon, hdg, spd = struct.unpack_from(">ddff", data, meta_off)
        self._emit(SonarPing(
            port_data=port, starboard_data=stbd,
            latitude=lat or None, longitude=lon or None,
            heading_deg=hdg, speed_knots=spd,
        ))


# ─── XTF Reader ───────────────────────────────────────────────────────────────

class XTFSonarReader(SonarReader):
    """
    XTF (eXtended Triton Format) sonar file/stream reader.

    XTF is the industry-standard format for SSS recordings.
    Supports both:
      - Offline playback of .xtf files
      - Real-time streaming from a live XTF-writing sonar system

    XTF Packet Structure (simplified):
      FileHeader (1024 bytes) — read once at start
      Packets: [XTFPKTYPHDR 4B][Magic 0xFACE 2B][Packet body]
    """

    XTF_MAGIC = 0xFACE
    CHANINFO_SIZE = 128
    HEADER_SIZE = 1024

    def __init__(self, path: str, loop: bool = False,
                 playback_speed: float = 1.0, **kwargs):
        """
        Args:
            path:           Path to .xtf file.
            loop:           Restart when file ends (useful for testing).
            playback_speed: 1.0 = real-time, 0 = as fast as possible.
        """
        super().__init__(**kwargs)
        self.path = Path(path)
        self.loop = loop
        self.playback_speed = playback_speed

    def _read_loop(self):
        if not self.path.exists():
            logger.error(f"XTF file not found: {self.path}")
            return

        logger.info(f"Playing XTF file: {self.path} (loop={self.loop})")
        while self._running:
            self._play_file()
            if not self.loop or not self._running:
                break
            logger.info("XTF file ended — looping")

    def _play_file(self):
        with open(self.path, "rb") as f:
            # Skip file header
            header = f.read(self.HEADER_SIZE)
            if len(header) < self.HEADER_SIZE:
                logger.error("XTF file too small")
                return

            num_channels = struct.unpack_from("<H", header, 28)[0]
            samples_per_channel = {}

            # Read channel info headers
            for ch in range(min(num_channels, 6)):
                ch_offset = 64 + ch * self.CHANINFO_SIZE
                if ch_offset + self.CHANINFO_SIZE > len(header):
                    break
                samples = struct.unpack_from("<H", header, ch_offset + 8)[0]
                samples_per_channel[ch] = samples

            n_samples = samples_per_channel.get(0, 500)

            prev_ts = None
            while self._running:
                # Read packet header (256 bytes)
                pkt_header = f.read(256)
                if len(pkt_header) < 256:
                    break  # EOF

                magic = struct.unpack_from("<H", pkt_header, 0)[0]
                pkt_type = struct.unpack_from("<B", pkt_header, 2)[0]

                if magic != self.XTF_MAGIC:
                    # Re-sync: scan for magic
                    f.seek(-255, 1)
                    continue

                num_channels_pkt = struct.unpack_from("<H", pkt_header, 4)[0]
                byte_count = struct.unpack_from("<I", pkt_header, 6)[0]

                # Parse GPS fields from header
                lat  = struct.unpack_from("<d", pkt_header, 104)[0]
                lon  = struct.unpack_from("<d", pkt_header, 112)[0]
                hdg  = struct.unpack_from("<f", pkt_header, 120)[0]
                spd  = struct.unpack_from("<f", pkt_header, 128)[0]

                ping_time = struct.unpack_from("<d", pkt_header, 136)[0]

                # Read channel data
                channel_data = f.read(byte_count - 256) if byte_count > 256 else b""

                if pkt_type == 0 and num_channels_pkt >= 2:  # Sonar ping packet
                    half = len(channel_data) // 2
                    if half < 2:
                        continue

                    port_raw = np.frombuffer(channel_data[:half], dtype=np.uint8).astype(np.float32)
                    stbd_raw = np.frombuffer(channel_data[half:], dtype=np.uint8).astype(np.float32)

                    # Throttle to playback speed
                    if self.playback_speed > 0 and prev_ts is not None:
                        dt = ping_time - prev_ts
                        if 0 < dt < 1.0:
                            time.sleep(dt / self.playback_speed)
                    prev_ts = ping_time

                    self._emit(SonarPing(
                        port_data=port_raw,
                        starboard_data=stbd_raw,
                        timestamp=ping_time or time.time(),
                        latitude=lat if abs(lat) > 1e-6 else None,
                        longitude=lon if abs(lon) > 1e-6 else None,
                        heading_deg=hdg,
                        speed_knots=spd,
                    ))


# ─── File / Image Reader ──────────────────────────────────────────────────────

class FileSonarReader(SonarReader):
    """
    Replay sonar images from PNG/TIFF/JPG files as a simulated ping stream.

    Each image file is sliced into horizontal rows, with each row treated
    as one ping. Useful for:
      - Development without physical sonar hardware
      - Testing the full pipeline with known images
      - Benchmarking throughput

    Args:
        paths:          List of image file paths (or a directory).
        fps:            Target playback rate (rows/sec → approx pings/sec).
        loop:           Restart when all files are exhausted.
    """

    def __init__(self, paths, fps: float = 50.0, loop: bool = False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            if p.is_dir():
                exts = {".png", ".tiff", ".tif", ".jpg", ".jpeg", ".bmp"}
                self.paths = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
            else:
                self.paths = [p]
        else:
            self.paths = [Path(p) for p in paths]
        self.fps = fps
        self.loop = loop

    def _read_loop(self):
        if not self.paths:
            logger.error("FileSonarReader: no image files provided")
            return

        logger.info(f"FileSonarReader: replaying {len(self.paths)} file(s) at {self.fps} fps")
        delay = 1.0 / max(self.fps, 0.1)

        while self._running:
            for path in self.paths:
                if not self._running:
                    return
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Cannot read: {path}")
                    continue

                h, w = img.shape
                half_w = w // 2

                for row in range(h):
                    if not self._running:
                        return
                    port_row = img[row, :half_w].astype(np.float32)
                    stbd_row = img[row, half_w:].astype(np.float32)
                    self._emit(SonarPing(
                        port_data=port_row,
                        starboard_data=stbd_row,
                        timestamp=time.time(),
                    ))
                    time.sleep(delay)

            if not self.loop:
                logger.info("FileSonarReader: finished replaying all files")
                break
            logger.info("FileSonarReader: looping")


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_reader(interface: str, **kwargs) -> SonarReader:
    """
    Factory function — create a reader by interface name.

    Args:
        interface: One of "serial", "udp", "xtf", "file"
        **kwargs:  Passed directly to the reader constructor.
    """
    readers = {
        "serial": SerialSonarReader,
        "udp":    UDPSonarReader,
        "xtf":    XTFSonarReader,
        "file":   FileSonarReader,
    }
    cls = readers.get(interface.lower())
    if cls is None:
        raise ValueError(
            f"Unknown interface '{interface}'. "
            f"Choose from: {list(readers.keys())}"
        )
    return cls(**kwargs)
