# RepNCSP-YOLO：面向无人机影像小目标检测的高分辨率语义增强网络

> **RepNCSP-YOLO: a high-resolution semantic enhancement network for small object detection in UAV imagery**
>
> Yuhan Qi, Yutong Cao, Jianheng Zheng, **Chengjie Lu**, Yifan Yin, Zhuocheng Huang, WeiQi Hu, Li Shuai, Jialan Wu, Guangfa Tang\*
>
> School of Computing and Artificial Intelligence, Guangzhou Xinhua University, Dongguan, Guangdong, China
>
> 关键词：无人机影像 · 小目标检测 · YOLO26 · 高分辨率特征增强 · 结构重参数化 · VisDrone2019

---

## 目录

1. [项目简介](#1-项目简介)
2. [方法概述](#2-方法概述)
3. [仓库文件结构](#3-仓库文件结构)
4. [环境配置](#4-环境配置)
5. [数据集准备](#5-数据集准备-visdrone2019)
6. [代码改动（复现关键）](#6-代码改动复现关键)
7. [训练](#7-训练)
8. [验证 / 预测 / 导出](#8-验证--预测--导出)
9. [实验结果](#9-实验结果)
10. [常见问题与注意事项](#10-常见问题与注意事项)
11. [引用与许可](#11-引用与许可)

---

## 1. 项目简介

无人机航拍影像中的目标普遍**尺度小、分布密、遮挡多、背景复杂**。目标在输入图中往往只占十几个甚至几个像素，而常规 YOLO 检测器仅在 P3–P5 三个尺度上检测，反复下采样后小目标的边缘、纹理等空间信息被严重削弱。

本文以 **YOLO26n** 为基线，提出 **RepNCSP-YOLO**，包含两项核心改进：

1. **P2 高分辨率特征路径（P2 High-Resolution Path）**：在检测金字塔中显式复用 stride = 4 的 P2 浅层特征图（640×640 输入下分辨率为 **160×160**），为小目标保留细粒度空间细节，检测头扩展为 P2/P3/P4/P5 四尺度。
2. **RepNCSP 模块（Reparameterized CSP）**：在 P2 路径与主干/颈部以 RepNCSP 替换 C3k2。训练时每个 RepConv 包含 3×3 卷积、1×1 卷积、Identity 三个分支，获得更强的多分支特征表达；推理时可等价融合为单路卷积，兼顾训练期表达能力与部署期效率。

在 VisDrone2019 验证集上，RepNCSP-YOLO 取得所有参评模型中**最高的 Precision（0.498）、mAP@0.5（0.379）和 mAP@0.5:0.95（0.227）**；相比 YOLO26n 基线，mAP@0.5 与 mAP@0.5:0.95 分别相对提升 **8.0%** 和 **13.5%**。

---

## 2. 方法概述

![RepNCSP-YOLO 整体架构](assets/fig1_architecture.png)

*图 1：RepNCSP-YOLO 整体架构。红色虚线框为本文提出的 P2 高分辨率路径，其中红色模块为 RepNCSP 模块。*

### 2.1 高分辨率特征保留的动机

640×640 输入下，各检测层的特征图分辨率如下：

| 特征层 | 下采样倍率 | 特征图尺寸 | 每像素对应输入区域 | 单图网格数 |
|:---:|:---:|:---:|:---:|:---:|
| P2 | 4 (stride=4) | 160 × 160 | 4 × 4 | 25,600 |
| P3 | 8 (stride=8) | 80 × 80 | 8 × 8 | 6,400 |
| P4 | 16 (stride=16) | 40 × 40 | 16 × 16 | 1,600 |
| P5 | 32 (stride=32) | 20 × 20 | 32 × 32 | 400 |

一个 8×8 像素的微小目标，在 P2 上理论响应面积为 **4 个像素**，而在 P5 上仅为 **0.0625 个像素**（小于一个网格）。仅依赖深层特征必然丢失大量小目标，因此需要把 P2 高分辨率特征显式纳入特征融合，并通过自顶向下的上采样向其注入深层语义。

### 2.2 RepNCSP 模块

RepNCSP 采用 CSP（Cross Stage Partial）思想：输入经 1×1 卷积分成两个分支，一分支送入堆叠的 **RepBottleneck**（内部卷积均为 RepConv），另一分支直接保留，最后拼接并经 1×1 卷积融合。只对部分通道做增强，避免在 160×160 的大图上引入过大开销。

- **训练态**：RepConv = 3×3 Conv 分支 + 1×1 Conv 分支 + Identity(BN) 分支，三路相加后接 SiLU，多分支结构带来更强的特征学习能力。
- **推理态**：三条分支按卷积/BN 融合规则等价转换为一个 3×3 卷积，数学上完全等价，无额外推理开销。

### 2.3 网络配置

模型配置文件为 [`models/yolo26-p2-repncsp.yaml`](models/yolo26-p2-repncsp.yaml)，共 30 个层定义，结构要点：

- 主干沿用 YOLO26n：Conv → C3k2 → … → SPPF → C2PSA；
- 主干第 4/6/8 层（P3/P4/P5 阶段）与颈部全部特征融合块由 C3k2 替换为 **RepNCSP**；
- 与小目标最贴近的 P2 边缘层（主干第 2 层、颈部 19 层）**保留轻量 C3k2**，控制计算量；
- 颈部新增一次上采样并与主干 P2 拼接，形成 P2 高分辨率路径；
- 检测头 `Detect(P2, P3, P4, P5)`，输出步长 `[4, 8, 16, 32]`。

---

## 3. 仓库文件结构

```text
RepNCSP-YOLO/
├── README.md                       # 本文件
├── patch_repncsp.py                # （可选）一键给已安装的 ultralytics 打补丁脚本
├── models/
│   └── yolo26-p2-repncsp.yaml      # 本文网络结构配置（训练时 model 参数）
├── assets/
│   ├── fig1_architecture.png       # 网络结构图（论文 Fig.1）
│   ├── fig5_detection.png          # 定性检测对比（论文 Fig.5）
│   └── fig7_loss.png               # 训练损失收敛曲线（论文 Fig.7）
└── VisDrone2019/yolo_format/       # 数据集（需自行下载，见第 5 节）
    ├── VisDrone.yaml
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

实现层面涉及的文件（均以 ultralytics 8.4.10 为基准）：

| 文件 | 作用 | 相对原版的改动 |
|---|---|---|
| `ultralytics/nn/modules/block.py` | 网络模块定义 | **文件末尾追加** `RepConv`、`RepBottleneck`、`RepNCSP` 三个类 |
| `ultralytics/nn/tasks.py` | 模型解析 | **3 处**：导入 1 处 + 模块注册集合 2 处 |
| `ultralytics/nn/modules/__init__.py` | 模块导出 | **2 处**（导入与 `__all__` 各 1 处，**论文包未提供但必须改**） |
| `ultralytics/nn/modules/conv.py` | 卷积模块 | **无需改动**（与官方一致） |
| `models/yolo26-p2-repncsp.yaml` | 模型配置 | 新增文件，直接使用 |

---

## 4. 环境配置

### 4.1 硬件

- 论文实验平台：**NVIDIA GeForce RTX 4060**（本文实测环境为 RTX 4050 Laptop，6 GB 显存，`batch=8` 可正常训练；显存不足请按 2 的倍数下调 batch）。
- 训练/评估输入分辨率统一为 **640 × 640**。

### 4.2 软件版本（已验证）

| 依赖 | 版本 |
|---|---|
| OS | Windows 10/11（x64） |
| Python | 3.10.19 |
| PyTorch | 2.0.0 + CUDA 11.8（`torch==2.0.0+cu118`） |
| ultralytics | **8.4.10**（改动基于该版本，勿混用其他版本） |
| CUDA Driver | 与 cu118 兼容即可（≥ 522） |

### 4.3 安装命令

```powershell
# 1. 创建并激活 conda 环境
conda create -n yolo26 python=3.10 -y
conda activate yolo26

# 2. 安装 PyTorch 2.0.0 + CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 ultralytics 8.4.10（必须锁定版本）
pip install ultralytics==8.4.10

# 4. 验证
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
# 期望输出：2.0.0+cu118 True 8.4.10
```

---

## 5. 数据集准备 (VisDrone2019)

### 5.1 数据集说明

VisDrone2019-DET 包含 **10,000+ 张**城市/交通场景航拍图，训练集 6,471 张、验证集 548 张、test-dev 1,610 张，共 10 个类别：

| id | 类别 | id | 类别 |
|:---:|---|:---:|---|
| 0 | pedestrian（行人） | 5 | truck（卡车） |
| 1 | people（人员） | 6 | tricycle（三轮摩托） |
| 2 | bicycle（自行车） | 7 | awning-tricycle（遮阳三轮） |
| 3 | car（轿车） | 8 | bus（公交） |
| 4 | van（面包车） | 9 | motor（摩托车） |

下载地址：<https://github.com/VisDrone/VisDrone-Dataset>（需下载 `VisDrone2019-DET-train/val/test-dev` 三个压缩包）。

### 5.2 目录结构

将官方标注（`bbox_left, bbox_top, bbox_w, bbox_h, score, category, …`，逗号分隔）转换为 YOLO 格式（归一化的 `class x_center y_center w h`，忽略 `score=0` 的区域，类别 id 减 1），并组织成：

```text
VisDrone2019/
└── yolo_format/
    ├── VisDrone.yaml
    ├── images/
    │   ├── train/   # 6471 张 .jpg
    │   ├── val/     # 548 张 .jpg
    │   └── test/    # 1610 张 .jpg
    └── labels/
        ├── train/   # 与图片同名的 .txt
        ├── val/
        └── test/
```

`VisDrone.yaml` 内容（与训练记录中的路径一致）：

```yaml
# VisDrone2019/yolo_format/VisDrone.yaml
path: ./VisDrone2019/yolo_format   # 数据集根目录（建议写绝对路径）
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

> 提示：也可以直接用 ultralytics 自带的 `data=VisDrone.yaml`（首次运行会自动下载并转换标注），但自动转换的目录名是 `VisDrone/`，与本文命令中的路径不同，二选一即可。

### 5.3 手动转换脚本（如使用官方原始压缩包）

```python
# convert_visdrone.py
from pathlib import Path
from PIL import Image

def visdrone2yolo(root: Path, src_split: str, dst_split: str):
    """src_split: 官方压缩包名后缀；dst_split: 转换后 images/labels 下的目录名。"""
    src = root / f"VisDrone2019-DET-{src_split}"
    (root / "images" / dst_split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
    for img in (src / "images").glob("*.jpg"):
        img.rename(root / "images" / dst_split / img.name)
    for f in (src / "annotations").glob("*.txt"):
        w, h = Image.open(root / "images" / dst_split / f.with_suffix(".jpg").name).size
        lines = []
        for row in [x.split(",") for x in f.read_text().strip().splitlines()]:
            if row[4] != "0":  # 跳过忽略区域
                continue
            x, y, bw, bh = map(int, row[:4])
            cls = int(row[5]) - 1
            lines.append(f"{cls} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")
        (root / "labels" / dst_split / f.name).write_text("".join(lines))

root = Path("VisDrone2019/yolo_format")
for src_split, dst_split in [("train", "train"), ("val", "val"), ("test-dev", "test")]:
    visdrone2yolo(root, src_split, dst_split)
```

---

## 6. 代码改动（复现关键）

首先定位 ultralytics 安装位置：

```powershell
python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))"
# 例如：D:\software\programming_software\anaconda3\envs\yolo26\Lib\site-packages\ultralytics
```

下文用 `<ULTRA>` 代指该目录。**推荐先运行一键补丁脚本**（会自动备份原文件为 `.bak`，且可重复执行）：

```powershell
python patch_repncsp.py
```

脚本会自动完成下面 6.1–6.3 的全部修改，并做一次构建自检。如果想手动修改，请按下面的步骤操作。

### 6.1 修改 `<ULTRA>/nn/modules/block.py`（末尾追加 3 个类）

在文件**最末尾**追加以下代码（不要删除原有内容）：

```python
# ==================== RepNCSP (RepConv + RepBottleneck + CSP) ====================
import torch
import torch.nn as nn
from .conv import Conv

class RepConv(nn.Module):
    # 重参数化卷积块 (训练时多分支，推理时可融合)
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        # 分支1: 正常的 3x3 卷积
        self.rbr_dense = Conv(c1, c2, k, s, p=p, g=g, act=False)
        # 分支2: 1x1 卷积
        self.rbr_1x1 = Conv(c1, c2, 1, s, p=0, g=g, act=False) if k != 1 else None
        # 分支3: Identity (恒等映射)
        self.rbr_identity = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 训练前向传播：三个分支相加
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        x1 = self.rbr_dense(x)
        x2 = 0 if self.rbr_1x1 is None else self.rbr_1x1(x)
        return self.act(x1 + x2 + id_out)

class RepBottleneck(nn.Module):
    # 使用 RepConv 的瓶颈层
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, 3, 1)
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class RepNCSP(nn.Module):
    # 结合了重参数化和 CSP 结构的最终模块 (直接替换 C3k2)
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # output conv
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

### 6.2 修改 `<ULTRA>/nn/tasks.py`（3 处）

**第 1 处** — 顶部 `from ultralytics.nn.modules import (...)` 导入块中，在 `C3k2,` 之后加一行：

```python
    C3k2,
    RepNCSP,   # 新增
    C3x,
```

**第 2 处** — `parse_model()` 内的 `base_modules = frozenset({...})` 中，在 `C3k2,` 之后加一行：

```python
            C3k2,
            RepNCSP,   # 新增
            RepNCSPELAN4,
```

**第 3 处** — 同函数内的 `repeat_modules = frozenset({...})` 中，在 `C3k2,` 之后加一行：

```python
            C3k2,
            RepNCSP,   # 新增
            C2fAttn,
```

### 6.3 修改 `<ULTRA>/nn/modules/__init__.py`（2 处，必须改）

`tasks.py` 是通过 `ultralytics.nn.modules` 包导入 RepNCSP 的，因此还需要在包导出中注册：

**第 1 处** — `from .block import (...)` 块中：

```python
    RepC3,
    RepNCSP,        # 新增
    RepNCSPELAN4,
```

**第 2 处** — 文件末尾的 `__all__ = (...)` 中：

```python
    "RepC3",
    "RepConv",
    "RepNCSP",       # 新增
    "RepNCSPELAN4",
```

### 6.4 放置模型配置

把 `models/yolo26-p2-repncsp.yaml` 放在工程目录下，训练时直接传该文件的相对/绝对路径即可（推荐保持与本仓库一致的 `models/` 目录结构）。

### 6.5 构建自检

```powershell
python -c "from ultralytics import YOLO; m=YOLO('models/yolo26-p2-repncsp.yaml'); m.info(imgsz=640)"
```

成功时应输出 **309 layers、约 3.10M parameters、约 12.3 GFLOPs（nc=10 时）**，检测头步长为 `[4, 8, 16, 32]`。若报 `ImportError: cannot import name 'RepNCSP'`，说明 6.3 的 `__init__.py` 没有改。

---

## 7. 训练

### 7.1 训练超参数（论文设置）

| 超参数 | 取值 |
|---|---|
| epochs | 200 |
| batch | 8 |
| imgsz | 640 |
| optimizer | SGD（momentum） |
| lr0 | 0.01 |
| lrf | 0.01（最终学习率 = lr0 × lrf = 1e-4） |
| momentum | 0.937 |
| weight_decay | 5e-4 |
| warmup_epochs | 3 |
| patience | 100（允许 100 轮无提升后早停） |
| 数据增强 | 随机水平翻转、颜色扰动（HSV）、Mosaic、Mixup |
| 评估分辨率 | 严格 640 × 640，不使用更大输入 |

### 7.2 训练本文模型（命令行）

```powershell
yolo detect train model=models/yolo26-p2-repncsp.yaml ^
  data=VisDrone2019/yolo_format/VisDrone.yaml ^
  epochs=200 batch=8 imgsz=640 device=0 workers=0 ^
  optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 ^
  warmup_epochs=3 patience=100 ^
  mosaic=1.0 mixup=0.1 fliplr=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 ^
  project=runs/detect name=RepNCSP-YOLO
```

> 说明：
> - 论文方法部分写明使用 Mosaic 与 Mixup；若需严格复现作者机器上的实际运行记录（检查点元数据为 `optimizer=auto, mixup=0.0`，ultralytics 8.4.10 在迭代数 > 10000 时 auto 会选择 SGD 系优化器），可将命令改为 `optimizer=auto` 并去掉 `mixup=0.1`，两种配置差异很小。
> - 显存不足时把 `batch=8` 调为 4/2，或加 `cache=False`；ultralytics 会自动按累积梯度保持等效 batch。
> - 若希望严格按论文文字（SGD + 上述全部增强），请使用上面的完整命令。

### 7.3 Python API 方式

```python
from ultralytics import YOLO

model = YOLO("models/yolo26-p2-repncsp.yaml")
model.train(
    data="VisDrone2019/yolo_format/VisDrone.yaml",
    epochs=200, batch=8, imgsz=640, device=0, workers=0,
    optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3, patience=100,
    mosaic=1.0, mixup=0.1, fliplr=0.5,
    project="runs/detect", name="RepNCSP-YOLO",
)
```

### 7.4 训练基线与消融模型

```powershell
# 基线 YOLO26n（P3-P5，无任何改动）
yolo detect train model=yolo26n.yaml data=VisDrone2019/yolo_format/VisDrone.yaml epochs=200 batch=8 imgsz=640 optimizer=SGD lr0=0.01 name=baseline-yolo26n

# 仅加 P2 路径（官方 yolo26-p2.yaml）
yolo detect train model=yolo26-p2.yaml data=VisDrone2019/yolo_format/VisDrone.yaml epochs=200 batch=8 imgsz=640 optimizer=SGD lr0=0.01 name=baseline-yolo26n-p2
```

论文中的 P2+EMA、P2+SPD 消融模型：在 `yolo26-p2.yaml` 基础上，分别把对应颈部模块替换为 EMA 注意力模块或 SPD-Conv 模块，训练参数与上表完全一致。

### 7.5 训练产物

训练结束后在 `runs/detect/RepNCSP-YOLO/` 下生成：

- `weights/best.pt` / `last.pt`（最佳与最后一轮权重）
- `results.csv`、`results.png`（各项指标曲线）
- `confusion_matrix.png`、`PR_curve.png`、`P_curve.png`、`R_curve.png`
- `train_batch*.jpg`（增强可视化）、`val_batch*_pred.jpg`（验证预测图）

---

## 8. 验证 / 预测 / 导出

```powershell
# 1) 在验证集上评估 Precision / Recall / mAP@0.5 / mAP@0.5:0.95
yolo detect val model=runs/detect/RepNCSP-YOLO/weights/best.pt ^
  data=VisDrone2019/yolo_format/VisDrone.yaml imgsz=640 batch=8 device=0

# 2) 对图片/文件夹/视频推理
yolo detect predict model=runs/detect/RepNCSP-YOLO/weights/best.pt ^
  source=path/to/images imgsz=640 conf=0.25 save=True

# 3) 导出 ONNX（训练态多分支融合为推理态单路卷积）
yolo export model=runs/detect/RepNCSP-YOLO/weights/best.pt imgsz=640 format=onnx simplify=True
```

---

## 9. 实验结果

> 以下指标全部来自论文（VisDrone2019 **验证集**，640×640，200 epochs，RTX 4060）。指标定义：Precision、Recall、mAP@0.5、mAP@0.5:0.95；其中 mAP@0.5:0.95 因定位要求更严格，作为主指标。

### 9.1 与主流轻量检测器 / UAV 专用检测器对比（论文 Table 1）

| 方法 | Recall↑ | Precision↑ | mAP@0.5↑ | mAP@0.5:0.95↑ |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.325 | 0.453 | 0.335 | 0.194 |
| YOLOv9t | 0.345 | 0.450 | 0.347 | 0.202 |
| YOLOv10n | 0.337 | 0.450 | 0.339 | 0.199 |
| YOLOv11n | 0.337 | 0.442 | 0.336 | 0.195 |
| YOLOv12n | 0.343 | 0.443 | 0.338 | 0.196 |
| YOLO26n（基线） | 0.344 | 0.456 | 0.351 | 0.200 |
| LPAE-YOLOv8 | 0.359 | 0.458 | 0.358 | 0.213 |
| SSCW-YOLO | **0.385** | 0.469 | 0.378 | 0.223 |
| DPCNet | 0.303 | 0.408 | 0.285 | 0.162 |
| YOLO-RECAP | 0.321 | 0.418 | 0.316 | 0.184 |
| **Ours（RepNCSP-YOLO）** | 0.367 | **0.498** | **0.379** | **0.227** |

**结论：**

- Precision、mAP@0.5、mAP@0.5:0.95 三项均为全部参评模型最高；Recall 略低于 SSCW-YOLO（0.367 vs 0.385），但精度显著更优，整体平衡更好。
- 相对 YOLO26n 基线：Precision 0.456 → 0.498（+0.042），mAP@0.5 0.351 → 0.379（**相对 +8.0%**），mAP@0.5:0.95 0.200 → 0.227（**相对 +13.5%**）。
- 严格 IoU 阈值下提升更大，说明 P2 路径带来的精细定位能力与 RepNCSP 的判别增强真实有效。

### 9.2 消融实验（论文 Table 2）

| 配置 | Recall↑ | Precision↑ | mAP@0.5↑ | mAP@0.5:0.95↑ |
|---|---:|---:|---:|---:|
| YOLO26n（基线） | 0.344 | 0.456 | 0.351 | 0.200 |
| + P2 高分辨率路径 | 0.351 | 0.456 | 0.358 | 0.214 |
| + P2 + EMA | 0.355 | 0.481 | 0.367 | 0.220 |
| + P2 + SPD | 0.368 | 0.450 | 0.365 | 0.217 |
| **+ P2 + RepNCSP（Ours）** | **0.367** | **0.498** | **0.379** | **0.227** |

**结论：**

- 仅加 P2 路径，mAP@0.5:0.95 即从 0.200 提升到 0.214，验证了浅层高分辨率特征对小目标定位的价值；
- 在 P2 上对比三种增强策略，EMA（mAP@0.5:0.95 0.220）、SPD（0.217）均有效，但 **RepNCSP 综合最优（0.227）**，且 Precision 大幅领先（0.498）。

### 9.3 定性检测结果

![定性检测对比](assets/fig5_detection.png)

*图 2（论文 Fig.5）：复杂背景、极小目标、密集目标三类典型场景下的检测对比。黄色框标注基线误检/漏检区域，RepNCSP-YOLO 显著减少误检与漏检，密集场景下实例分离更清晰。*

Grad-CAM 热力图（论文 Fig.6）进一步表明：基线在复杂背景上激活分散、噪声大，而 RepNCSP-YOLO 的激活集中在车辆与行人目标上；在极小目标和密集簇上激活峰更尖锐，说明 P2 路径与 RepNCSP 有效抑制了背景干扰。

### 9.4 训练收敛性

![训练损失曲线](assets/fig7_loss.png)

*图 3（论文 Fig.7）：200 轮训练中边界框回归损失（a）与分类损失（b）曲线。RepNCSP-YOLO（紫色）早期下降更快，框回归损失稳定在更低的平衡点，分类损失更平滑、波动更小。*

### 9.5 模型规模实测（本次环境构建实测，供参考，非论文数据）

在 nc=10、imgsz=640 下用本仓库配置实际构建统计：

| 模型 | 层数 | 参数量 | GFLOPs@640 | 检测步长 |
|---|---:|---:|---:|:---:|
| YOLO26n（官方） | 260 | 2,507,700 | 5.8 | 8/16/32 |
| YOLO26n-P2（官方） | 329 | 2,518,352 | 7.5 | 4/8/16/32 |
| **RepNCSP-YOLO（本文）** | **309** | **3,103,512** | **12.3** | **4/8/16/32** |

- P2 四尺度头在 640 输入下每图产生 25,600 + 6,400 + 1,600 + 400 = **34,000 个预测位置**（基线仅 8,400 个），这是 GFLOPs 增加的主要来源，也是检测小目标的必要代价；
- 本文模型参数约 3.10M（较基线 +约 0.6M），仍属轻量级，训练态多分支结构是计算量上升的主因，**部署时经卷积/BN 融合可明显降低推理开销**（论文亦将该部分列为后续实测工作）。

---

## 10. 常见问题与注意事项

1. **`ImportError: cannot import name 'RepNCSP'`**
   只改了 `block.py` / `tasks.py` 还不够，必须按 6.3 修改 `ultralytics/nn/modules/__init__.py` 的导入与 `__all__`。

2. **关于版本**
   改动基于 **ultralytics 8.4.10**（对应 torch 2.0.0+cu118）。其他版本的 `tasks.py` 结构可能不同，一键脚本会校验版本；跨版本请参照 6.1–6.3 的位置手动改。

3. **关于 `end2end` / `reg_max`**
   官方 `yolo26.yaml` 默认带 `end2end: True`、`reg_max: 1`（免 NMS 端到端头）。本配置**未启用**这两项，使用常规 Detect 头 + DFL（reg_max=16），训练输出与论文一致；推理时走标准 NMS。如想启用端到端模式需自行在 yaml 顶部加回这两行并重新训练，指标不可直接比较。

4. **yaml 中的 `nc: 80`**
   配置文件里写 80 只是占位，训练时会被数据集 yaml 的 10 类自动覆盖（日志可见 `Overriding model.yaml nc=80 with nc=10`），无需手动改。

5. **随附 `best.pt` 的说明**
   本次提供的 `best.pt` 经检查是 **YOLOv8n 基线复现权重**（yolov8.yaml，3,012,798 参数；指标 P=0.453 / R=0.325 / mAP50=0.335 / mAP50-95=0.194，与论文 Table 1 的 YOLOv8n 行一致），并非 RepNCSP-YOLO 的最终权重。RepNCSP-YOLO 权重请按第 7 节训练后获得 `runs/detect/RepNCSP-YOLO/weights/best.pt`。

6. **结果波动**
   不同显卡/驱动、随机种子下，mAP 末位可能有 ±0.002~0.005 的波动；复现时以整套训练配置（200 epochs、SGD、640、相同增强）为准。

7. **推理加速**
   部署前先 `yolo export ...` 或调用模型的 `fuse()`，将 RepConv 多分支及 Conv-BN 融合为单卷积，可在不损失精度的前提下加速。

---

## 11. 引用与许可

论文在投/见刊信息请以正式发表版本为准，引用格式示例（信息请按实际卷期补全）：

```bibtex
@article{qi2026repncsp,
  title={RepNCSP-YOLO: a high-resolution semantic enhancement network for small object detection in UAV imagery},
  author={Qi, Yuhan and Cao, Yutong and Zheng, Jianheng and Lu, Chengjie and Yin, Yifan and Huang, Zhuocheng and Hu, WeiQi and Shuai, Li and Wu, Jialan and Tang, Guangfa},
  journal={Manuscript},
  year={2026}
}
```

本工程基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0 License）修改，使用与分发请遵循其许可协议。数据集 VisDrone2019 版权归原作者所有。
