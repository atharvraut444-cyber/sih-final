#!/bin/bash
# ============================================================
# SENORITA — SSS Marine Debris Detection System Installer
# Supports: NVIDIA Jetson Orin/Nano, Raspberry Pi 4/5, Linux PC
# ============================================================
set -e

INSTALL_DIR="/opt/sonar_detection"
SERVICE_NAME="sonar_detection"
SERVICE_USER="sonar"
# On Linux/Jetson/RPi: python3 is correct.
# On Windows dev machines: set PYTHON="py -3" or use the python.bat shim.
PYTHON="${PYTHON:-python3}"

echo ""
echo "=== SENORITA — SSS Marine Debris Detection System Installer ==="
echo "     Target: $(uname -m) | $(cat /proc/device-tree/model 2>/dev/null || echo 'Linux PC')"
echo ""

# ── Create system user ────────────────────────────────────────
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "[1/7] Creating service user: $SERVICE_USER"
    useradd -r -s /bin/false -d $INSTALL_DIR $SERVICE_USER
    usermod -aG gpio,i2c,dialout $SERVICE_USER 2>/dev/null || true
fi

# ── Copy project files ────────────────────────────────────────
echo "[2/7] Installing to $INSTALL_DIR"
mkdir -p $INSTALL_DIR
cp -r . $INSTALL_DIR/
chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR

# ── Create data directories ───────────────────────────────────
echo "[3/7] Creating data directories"
mkdir -p $INSTALL_DIR/data/{images,reports,logs}
mkdir -p $INSTALL_DIR/uploads/results
mkdir -p $INSTALL_DIR/models
chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR/data
chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR/uploads

# ── Install Python dependencies ───────────────────────────────
echo "[4/7] Installing Python dependencies"
cd $INSTALL_DIR

# Edge-specific minimal requirements (no training stack)
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0 \
    aiofiles>=23.2.0 \
    websockets>=12.0

# Ultralytics ONNX runtime (lighter than full PyTorch on embedded)
$PYTHON -m pip install onnxruntime>=1.16.0

# Ultralytics for model loading (needed for .pt weights)
$PYTHON -m pip install ultralytics>=8.1.0 || \
    echo "WARNING: ultralytics install failed — demo mode only"

# Serial port support (for SSS hardware)
$PYTHON -m pip install pyserial>=3.5 || echo "WARNING: pyserial not installed"

echo "[4/7] Dependencies installed"

# ── Jetson-specific: TensorRT ──────────────────────────────────
if [[ $(cat /proc/device-tree/model 2>/dev/null) == *"Jetson"* ]]; then
    echo "[5/7] Jetson detected — checking TensorRT..."
    python3 -c "import tensorrt; print('TensorRT OK')" 2>/dev/null || \
        echo "       TensorRT not found — install via JetPack SDK for max performance"
else
    echo "[5/7] Non-Jetson platform — skipping TensorRT"
fi

# ── Install systemd service ───────────────────────────────────
echo "[6/7] Installing systemd service"
cp $INSTALL_DIR/deploy/sonar_detection.service /etc/systemd/system/${SERVICE_NAME}.service
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
echo "       Service enabled: systemctl start ${SERVICE_NAME}"

# ── Final status ──────────────────────────────────────────────
echo "[7/7] Installation complete!"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║     SSS Detection System — Ready                  ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║ Start service:   systemctl start ${SERVICE_NAME}   ║"
echo "║ View logs:       journalctl -fu ${SERVICE_NAME}    ║"
echo "║ Dashboard:       http://$(hostname -I | awk '{print $1}'):8000/ ║"
echo "║ API docs:        http://$(hostname -I | awk '{print $1}'):8000/docs ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "IMPORTANT: Edit /etc/systemd/system/${SERVICE_NAME}.service"
echo "           to set your SSS hardware interface (serial/UDP/XTF)"
echo ""
