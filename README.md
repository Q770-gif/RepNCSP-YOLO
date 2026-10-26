# RepNCSP-YOLO: A High-Resolution Semantic Enhancement Network for Small Object Detection in UAV Imagery

> **RepNCSP-YOLO: a high-resolution semantic enhancement network for small object detection in UAV imagery**
>
> Yuhan Qi, Yutong Cao, Jianheng Zheng, Chengjie Lu, Yifan Yin, Zhuocheng Huang, Weiqi Hu, Li Shuai, Jialan Wu, Guangfa Tang\*
>
> School of Computing and Artificial Intelligence, Guangzhou Xinhua University, Dongguan, Guangdong, China
>
> Keywords: UAV imagery · small object detection · YOLO26 · high-resolution feature enhancement · structural reparameterization · VisDrone2019

---

## Contents

1. [Introduction](#1-introduction)
2. [Method Overview](#2-method-overview)
3. [Environment](#3-environment)
4. [Dataset Preparation (VisDrone2019)](#4-dataset-preparation-visdrone2019)
5. [Code Modifications](#5-code-modifications)
6. [Training / Validation / Inference](#6-training--validation--inference)
7. [Experimental Results](#7-experimental-results)
8. [Notes and Caveats](#8-notes-and-caveats)

---

## 1. Introduction

Objects in drone-captured imagery are typically **small in scale, densely distributed, heavily occluded, and embedded in complex backgrounds**. A target may occupy only a dozen or even a few pixels in the input, while conventional YOLO detectors predict only at the P3–P5 scales; after repeated downsampling, the edge and texture spatial details of small objects are severely weakened.

Starting from the **YOLO26n** baseline, this paper proposes **RepNCSP-YOLO** with two core contributions:

1. **P2 High-Resolution Path**: explicitly reuses the shallow P2 feature map (stride = 4, resolution **160×160** for a 640×640 input) in the detection pyramid to preserve fine-grained spatial details for small objects; the detection head is extended to four scales, P2/P3/P4/P5.
2. **RepNCSP module (Reparameterized CSP)**: replaces C3k2 with RepNCSP in the backbone, neck, and P2 path. During training each RepConv contains three branches — a 3×3 convolution, a 1×1 convolution, and an Identity (BN) — for stronger multi-branch feature representation; at inference the branches can be equivalently fused into a single convolution, balancing training-time capacity with deployment efficiency.

On the VisDrone2019 validation set, RepNCSP-YOLO achieves the **highest Precision (0.498), mAP@0.5 (0.379), and mAP@0.5:0.95 (0.227)** among all compared models; relative to the YOLO26n baseline, mAP@0.5 and mAP@0.5:0.95 improve by **8.0%** and **13.5%**, respectively.

## 2. Method Overview

### 2.1 Motivation for Retaining High-Resolution Features

For a 640×640 input, the spatial resolution of each detection level is:

| Level | Downsample factor | Feature map size | Input area per pixel | Grid cells per image |
|:---:|:---:|:---:|:---:|:---:|
| P2 | 4 (stride=4) | 160 × 160 | 4 × 4 | 25,600 |
| P3 | 8 (stride=8) | 80 × 80 | 8 × 8 | 6,400 |
| P4 | 16 (stride=16) | 40 × 40 | 16 × 16 | 1,600 |
| P5 | 32 (stride=32) | 20 × 20 | 32 × 32 | 400 |

An 8×8-pixel tiny object theoretically covers **4 pixels** on P2 but only **0.0625 pixels** on P5 (less than one grid cell). Relying solely on deep features inevitably misses many small targets; the P2 high-resolution features must therefore be explicitly incorporated into feature fusion, with deep semantic information injected into them through top-down upsampling.

### 2.2 The RepNCSP Module

RepNCSP follows the CSP (Cross Stage Partial) idea: the input is split by a 1×1 convolution into two branches. One branch goes through stacked **RepBottlenecks** (whose internal convolutions are all RepConv), the other is retained directly, and the two are concatenated and fused by a 1×1 convolution. Enhancing only a subset of channels keeps the overhead on the large 160×160 map bounded.

- **Training mode**: RepConv = 3×3 Conv branch + 1×1 Conv branch + Identity(BN) branch; the three are summed and followed by SiLU. The multi-branch structure yields stronger feature learning.
- **Inference mode**: the three branches are equivalently converted into a single 3×3 convolution following the convolution/BN fusion rules — mathematically exact, with no extra inference cost.

### 2.3 Network Architecture (`yolo26-p2-repncsp.yaml`)

- The backbone follows YOLO26n: Conv → C3k2 → … → SPPF → C2PSA;
- The P3/P4/P5 stages in the backbone and the neck fusion blocks replace C3k2 with **RepNCSP**;
- The P2 edge layer closest to small targets (the shallow backbone block and the lightweight block before the P2 detection output) **keeps C3k2** to control compute;
- One extra upsampling step is added in the neck and concatenated with backbone P2, forming the P2 high-resolution path;
- The detection head is `Detect(P2, P3, P4, P5)` with output strides `[4, 8, 16, 32]`.

## 3. Environment

| Dependency | Version |
|---|---|
| OS | Windows / Linux (x64) |
| Python | 3.10 |
| PyTorch | 2.0.0 + CUDA 11.8 (`torch==2.0.0+cu118`) |
| ultralytics | **8.4.10** (modifications verified against this version) |
| Hardware | NVIDIA RTX 4060 (paper); training/evaluation input fixed at 640×640 |

```bash
conda create -n yolo26 python=3.10 -y && conda activate yolo26
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.4.10
```

## 4. Dataset Preparation (VisDrone2019)

VisDrone2019-DET contains **10,000+** aerial images of urban/traffic scenes (6,471 train, 548 val, 1,610 test-dev) across 10 classes:

| id | Class | id | Class |
|:---:|---|:---:|---|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

Download: <https://github.com/VisDrone/VisDrone-Dataset>. Convert the official comma-separated annotations to YOLO format (normalized `class x_center y_center w h`, drop `score=0` regions, subtract 1 from the class id), and organize the directory as:

```text
VisDrone2019/yolo_format/
├── VisDrone.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

`VisDrone.yaml`:

```yaml
path: ./VisDrone2019/yolo_format
train: images/train
val: images/val
test: images/test
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

## 5. Code Modifications

Modifications are based on ultralytics 8.4.10 and touch 3 files (`conv.py` requires no change):

1. **`ultralytics/nn/modules/block.py`**: append three classes at the end — `RepConv` (three-branch reparameterized convolution), `RepBottleneck` (its stacked bottleneck), and `RepNCSP` (CSP structure that directly replaces C3k2).
2. **`ultralytics/nn/tasks.py`**: 3 edits — add `RepNCSP` to the top import block, and to both the `base_modules` and `repeat_modules` frozensets in `parse_model()`.
3. **`ultralytics/nn/modules/__init__.py`**: add `RepNCSP` to the import block and to `__all__` (**required**, otherwise `ImportError`).

## 6. Training / Validation / Inference

### 6.1 Training Hyperparameters (paper settings)

| Hyperparameter | Value |
|---|---|
| epochs | 200 |
| batch | 8 |
| imgsz | 640 |
| optimizer | SGD (momentum 0.937) |
| lr0 / lrf | 0.01 / 0.01 (final lr 1e-4) |
| weight_decay | 5e-4 |
| warmup_epochs | 3 |
| patience | 100 |
| Augmentation | random horizontal flip, HSV color jitter, Mosaic, Mixup |

### 6.2 Training Command

```bash
yolo detect train model=models/yolo26-p2-repncsp.yaml \
  data=VisDrone2019/yolo_format/VisDrone.yaml \
  epochs=200 batch=8 imgsz=640 device=0 \
  optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3 patience=100 \
  mosaic=1.0 mixup=0.1 fliplr=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  project=runs/detect name=RepNCSP-YOLO
```

Python API:

```python
from ultralytics import YOLO
model = YOLO("models/yolo26-p2-repncsp.yaml")
model.train(
    data="VisDrone2019/yolo_format/VisDrone.yaml",
    epochs=200, batch=8, imgsz=640, device=0,
    optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, warmup_epochs=3, patience=100,
    mosaic=1.0, mixup=0.1, fliplr=0.5,
    project="runs/detect", name="RepNCSP-YOLO",
)
```

### 6.3 Baseline / Ablation / Evaluation

```bash
# Baseline YOLO26n
yolo detect train model=yolo26n.yaml data=VisDrone2019/yolo_format/VisDrone.yaml \
  epochs=200 batch=8 imgsz=640 optimizer=SGD lr0=0.01 name=baseline-yolo26n

# P2-only path (official yolo26-p2.yaml); P2+EMA and P2+SPD replace the corresponding
# enhancement module on top of it

# Evaluation
yolo detect val model=runs/detect/RepNCSP-YOLO/weights/best.pt \
  data=VisDrone2019/yolo_format/VisDrone.yaml imgsz=640 batch=8

# Inference / ONNX export
yolo detect predict model=runs/detect/RepNCSP-YOLO/weights/best.pt source=path/to/images imgsz=640 conf=0.25 save=True
yolo export model=runs/detect/RepNCSP-YOLO/weights/best.pt imgsz=640 format=onnx simplify=True
```

## 7. Experimental Results

> Metrics are on the VisDrone2019 **validation set**, 640×640, 200 epochs, RTX 4060; mAP@0.5:0.95 is the strict-location primary metric.

### 7.1 Comparison with mainstream lightweight / UAV-specific detectors (paper Table 1)

| Method | Recall↑ | Precision↑ | mAP@0.5↑ | mAP@0.5:0.95↑ |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.325 | 0.453 | 0.335 | 0.194 |
| YOLOv9t | 0.345 | 0.450 | 0.347 | 0.202 |
| YOLOv10n | 0.337 | 0.450 | 0.339 | 0.199 |
| YOLOv11n | 0.337 | 0.442 | 0.336 | 0.195 |
| YOLOv12n | 0.343 | 0.443 | 0.338 | 0.196 |
| YOLO26n (baseline) | 0.344 | 0.456 | 0.351 | 0.200 |
| LPAE-YOLOv8 | 0.359 | 0.458 | 0.358 | 0.213 |
| SSCW-YOLO | **0.385** | 0.469 | 0.378 | 0.223 |
| DPCNet | 0.303 | 0.408 | 0.285 | 0.162 |
| YOLO-RECAP | 0.321 | 0.418 | 0.316 | 0.184 |
| **Ours (RepNCSP-YOLO)** | 0.367 | **0.498** | **0.379** | **0.227** |

- Highest Precision, mAP@0.5, and mAP@0.5:0.95 among all compared models; Recall is slightly below SSCW-YOLO (0.367 vs 0.385) but precision is markedly higher, giving a better overall balance.
- Relative to the YOLO26n baseline: Precision 0.456 → 0.498, mAP@0.5 0.351 → 0.379 (**+8.0% relative**), mAP@0.5:0.95 0.200 → 0.227 (**+13.5% relative**).

### 7.2 Ablation study (paper Table 2)

| Configuration | Recall↑ | Precision↑ | mAP@0.5↑ | mAP@0.5:0.95↑ |
|---|---:|---:|---:|---:|
| YOLO26n (baseline) | 0.344 | 0.456 | 0.351 | 0.200 |
| + P2 high-resolution path | 0.351 | 0.456 | 0.358 | 0.214 |
| + P2 + EMA | 0.355 | 0.481 | 0.367 | 0.220 |
| + P2 + SPD | 0.368 | 0.450 | 0.365 | 0.217 |
| **+ P2 + RepNCSP (Ours)** | **0.367** | **0.498** | **0.379** | **0.227** |

- Adding only the P2 path raises mAP@0.5:0.95 from 0.200 to 0.214, confirming the value of shallow high-resolution features for small-target localization;
- Among the three enhancement strategies, EMA (0.220) and SPD (0.217) are effective, while **RepNCSP achieves the best overall result (0.227)** with a large Precision lead (0.498).

### 7.3 Qualitative and Convergence Analysis

- **Complex background / tiny objects / dense objects** (paper Fig. 5): RepNCSP-YOLO markedly reduces false positives and missed detections, and separates instances more cleanly in crowded scenes.
- **Grad-CAM heatmaps (Fig. 6)**: the baseline shows scattered, noisy activation; our method concentrates activation on the target, with sharper peaks on tiny objects and dense clusters, indicating that the P2 path and RepNCSP suppress background interference.
- **Training convergence (Fig. 7)**: over 200 epochs our method descends faster early, the box regression loss settles at a lower equilibrium, and the classification loss is smoother.

## 8. Notes and Caveats

1. Modifications are based on ultralytics **8.4.10**; the `tasks.py` layout may differ in other versions — adjust manually against the locations in Section 5.
2. `nc: 80` in the model yaml is only a placeholder; it is automatically overwritten by the 10 dataset classes at training time and needs no manual edit.
3. This configuration does **not** enable the official YOLO26 `end2end`/`reg_max` head; it uses the standard Detect head + DFL and conventional NMS at inference.
4. Before deployment, RepConv multi-branches can be equivalently compressed into a single convolution via convolution/BN fusion, with no accuracy loss and lower inference cost.

## Citation

```bibtex
@article{qi2026repncsp,
  title={RepNCSP-YOLO: a high-resolution semantic enhancement network for small object detection in UAV imagery},
  author={Qi, Yuhan and Cao, Yutong and Zheng, Jianheng and Lu, Chengjie and Yin, Yifan and Huang, Zhuocheng and Hu, WeiQi and Shuai, Li and Wu, Jialan and Tang, Guangfa},
  year={2026}
}
```

This project is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0 License); VisDrone2019 copyright belongs to its original authors.
