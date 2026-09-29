# TRAINING_GUIDE.md — YOLOv8 Sonar Detection Model Training & Fine-Tuning

> [!NOTE]
> A genuine 7-class sonar-trained model is active and deployed at `models/yolov8_sonar.pt`.
> Training was performed using Ultralytics YOLOv8s transfer learning with `imgsz=416`.
> This guide documents the active model configuration, dataset state, and instructions
> for expanding the dataset to eliminate undertrained class false alarms.

---

## 1. Active Model State & Provenance

* **Model Path**: `models/yolov8_sonar.pt`
* **Base Architecture**: YOLOv8s (Ultralytics)
* **Active Mode**: `detection_mode = "model"`
* **Current Training Parameters** (`runs/detect/training/runs/yolov8_sonar/args.yaml`):
  * Epochs: `3`
  * Input Resolution (`imgsz`): `416` (strictly aligned with serving config `config.py:MODEL_INPUT_SIZE = 416`)
  * Device: `cpu`
  * Batch Size: `8`
  * Pretrained: `true` (from `yolov8s.pt`)
* **Current Dataset Size**:
  * `train`: **60 images**, **190 annotations** across 7 classes
  * `val`: 20 images
  * `test`: 30 images
  * Total: 110 images

---

## 2. Dataset Limitations & False-Positive Root Causes

Investigation into benchmark performance revealed two key training limitations:

1. **Class Imbalance (Undertrained Classes)**:
   * Class 5 (`cable`) has only 18 annotations in the training split, resulting in poor feature discrimination (AP50 ~ 17%).
   * Class 3 (`cylinder`) similarly has limited training variety.
   * On high-roughness seabeds (coral and rock), background acoustic backscatter ripples trigger false-positive `cable` and `cylinder` detections.

2. **Absence of Negative Background Images**:
   * The training dataset contains 0 negative seafloor background images (images containing only coral, rock, or sand ripples without any debris).
   * In YOLO training, including 5%–10% background images (with empty `.txt` label files) is critical to teach the model what the bare seabed looks like and drastically reduce false alarms.

---

## 3. Recommended Dataset Expansion

To achieve production-grade precision (< 15% false alarm rate on rock/coral):

### Annotation Targets
- **Target Size**: $\ge 500$ annotations per class across all 7 classes.
- **Negative Background Images**: Add at least 100–200 empty seafloor patches (pure sand, ripple fields, jagged granite rock, coral outcrops) with empty annotation files.
- **Multi-Frequency Coverage**: Ensure balanced representation across 100 kHz, 300 kHz, 600 kHz, and 900 kHz imagery.

### Class Mapping (`config.py → CLASS_NAMES`)
| ID | Class Name | Description |
|:---|:-----------|:------------|
| 0 | ghost_net | Discarded fishing nets / ropes |
| 1 | shipwreck | Hull fragments, wooden/metal wrecks |
| 2 | pipe | Industrial conduits, drainage lines |
| 3 | cylinder | Barrels, fuel drums, gas cylinders |
| 4 | debris_cluster | Mixed scattered debris fields |
| 5 | cable | Underwater power/communication cables |
| 6 | anomaly | Unidentified acoustic anomalies |

### Directory Structure
```
data/
  sonar_dataset/
    images/
      train/   ← 60+ images (.png, .jpg, .tiff)
      val/     ← 20+ images
      test/    ← 30+ images
    labels/
      train/   ← YOLO-format .txt files (empty for negative backgrounds)
      val/
      test/
    dataset.yaml
```

---

## 4. Retraining Procedure

Run the training pipeline using the Python 3 environment:

```bash
# Full retraining on GPU (Jetson Orin or vessel workstation)
py -3 training/train.py \
  --dataset data/sonar_dataset/dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 416 \
  --device 0

# Retraining on CPU (development fallback)
py -3 training/train.py \
  --dataset data/sonar_dataset/dataset.yaml \
  --epochs 50 \
  --batch 8 \
  --imgsz 416 \
  --device cpu
```

### Deploying New Weights

1. Copy best weights to the models directory:
   ```bash
   cp runs/train/exp/weights/best.pt models/yolov8_sonar.pt
   ```

2. Verify model loading and run validation suite:
   ```bash
   py -3 tests/benchmark_ocean.py --n 5 --force
   ```

3. Export to ONNX for edge acceleration:
   ```bash
   py -3 training/export_model.py --weights models/yolov8_sonar.pt --imgsz 416
   ```

---

## 5. Domain Adaptation & Preprocessing

For optimal model performance across different sonar hardware:
* Apply Time-Varied Gain (TVG) normalization prior to annotation.
* Maintain preprocessor CLAHE and NLM filtering consistency between training and inference (`core/preprocessor.py`).
* Keep `imgsz=416` consistent across training, validation, export, and serving.
