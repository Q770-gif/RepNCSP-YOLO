# -*- coding: utf-8 -*-
"""
RepNCSP-YOLO 一键补丁脚本
=========================

作用：给当前 Python 环境中已安装的 ultralytics（基于 8.4.10 验证）打入
RepNCSP 相关改动，共 3 个文件：

  1. ultralytics/nn/modules/block.py        末尾追加 RepConv / RepBottleneck / RepNCSP
  2. ultralytics/nn/tasks.py                导入 + base_modules + repeat_modules 共 3 处
  3. ultralytics/nn/modules/__init__.py     导入 + __all__ 共 2 处

特性：
  - 幂等：已打过的补丁会自动跳过，可重复执行；
  - 安全：首次修改前自动备份为 *.bak（已存在备份则不覆盖）；
  - 自检：补丁完成后尝试用 models/yolo26-p2-repncsp.yaml 构建模型。

用法：
  python patch_repncsp.py                 # 打补丁并自检
  python patch_repncsp.py --no-build      # 只打补丁不构建模型
  python patch_repncsp.py --restore       # 从 .bak 还原全部文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_VERSION = "8.4.10"

# ---------------------------------------------------------------------------
# 1) block.py 末尾追加的代码
# ---------------------------------------------------------------------------
BLOCK_APPEND = '''
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
'''

# ---------------------------------------------------------------------------
# 2) 各文件的 (锚点, 替换后文本) 编辑项；使用 @NL@ 占位换行，应用时按文件实际换行替换
# ---------------------------------------------------------------------------
EDITS = {
    "nn/tasks.py": [
        # 顶部导入块
        ("    C3k2,@NL@    C3x,",
         "    C3k2,@NL@    RepNCSP,@NL@    C3x,"),
        # base_modules frozenset
        ("            C3k2,@NL@            RepNCSPELAN4,",
         "            C3k2,@NL@            RepNCSP,@NL@            RepNCSPELAN4,"),
        # repeat_modules frozenset
        ("            C3k2,@NL@            C2fAttn,",
         "            C3k2,@NL@            RepNCSP,@NL@            C2fAttn,"),
    ],
    "nn/modules/__init__.py": [
        # from .block import (...) 块
        ("    RepC3,@NL@    RepNCSPELAN4,",
         "    RepC3,@NL@    RepNCSP,@NL@    RepNCSPELAN4,"),
        # __all__ 元组
        ('    "RepC3",@NL@    "RepConv",@NL@    "RepNCSPELAN4",',
         '    "RepC3",@NL@    "RepConv",@NL@    "RepNCSP",@NL@    "RepNCSPELAN4",'),
    ],
}


def load(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    nl = "\r\n" if b"\r\n" in raw[:4000] else "\n"
    return raw.decode("utf-8"), nl


def save(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"  [备份] {path.name} -> {bak.name}")
    else:
        print(f"  [备份] {bak.name} 已存在，保留不覆盖")


def apply_edits(rel_path: str, root: Path) -> bool:
    path = root / rel_path
    text, nl = load(path)
    changed = False
    for anchor_t, replacement_t in EDITS[rel_path]:
        anchor = anchor_t.replace("@NL@", nl)
        replacement = replacement_t.replace("@NL@", nl)
        if replacement.split("@NL@")[0] and "RepNCSP" in text:
            pass
        # 幂等判断：替换目标文本若已存在则跳过
        if replacement in text:
            print(f"  [跳过] {rel_path} 中该补丁已存在")
            continue
        if anchor not in text:
            print(f"  [错误] 在 {rel_path} 中找不到锚点，可能版本不匹配：")
            print("         " + repr(anchor[:80]))
            return False
        text = text.replace(anchor, replacement, 1)
        changed = True
        print(f"  [应用] {rel_path} 锚点替换成功")
    if changed:
        backup(path)
        save(path, text)
    return True


def patch_block(root: Path) -> bool:
    path = root / "nn/modules/block.py"
    text, nl = load(path)
    # 注意：必须匹配 "class RepNCSP("，不能只搜 "class RepNCSP"，否则会误匹配
    # 原版已有的 RepNCSPELAN4
    if "class RepNCSP(" in text:
        print("  [跳过] block.py 已包含 RepNCSP 定义")
        return True
    backup(path)
    append = BLOCK_APPEND.replace("\n", nl)
    if not text.endswith(nl):
        text += nl
    text += append
    save(path, text)
    print("  [应用] block.py 末尾追加 RepConv / RepBottleneck / RepNCSP")
    return True


def restore(root: Path) -> None:
    for rel in ["nn/modules/block.py", "nn/tasks.py", "nn/modules/__init__.py"]:
        path = root / rel
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            path.write_bytes(bak.read_bytes())
            print(f"  [还原] {rel}")
        else:
            print(f"  [跳过] {rel} 无 .bak 备份")


def build_check(yaml_path: Path | None) -> bool:
    print("\n=== 构建自检 ===")
    try:
        import torch  # noqa: F401
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import RepNCSP  # noqa: F401
    except Exception as e:
        print(f"[失败] 导入异常: {e}")
        return False
    if yaml_path is None or not yaml_path.exists():
        print(f"[跳过] 未找到模型配置 {yaml_path}")
        return True
    try:
        model = DetectionModel(str(yaml_path), nc=10, verbose=False)
        # 注意：ultralytics 的 model.info 在 verbose=False 时返回 None，必须用默认 verbose=True
        n_l, n_p, n_g, flops = model.info(imgsz=640)
        print(f"[成功] 模型构建通过：{n_l} layers, {n_p:,} params, {flops:.1f} GFLOPs @640")
        head = model.model[-1]
        print(f"       检测头: {type(head).__name__}, strides={head.stride.tolist()}")
        return True
    except Exception as e:
        print(f"[失败] 模型构建出错: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="RepNCSP-YOLO ultralytics 补丁脚本")
    parser.add_argument("--no-build", action="store_true", help="只打补丁，不做模型构建自检")
    parser.add_argument("--restore", action="store_true", help="从 .bak 还原文件后退出")
    parser.add_argument("--yaml", default="models/yolo26-p2-repncsp.yaml", help="自检用模型配置路径")
    args = parser.parse_args()

    try:
        import ultralytics
    except ImportError:
        print("[错误] 未安装 ultralytics，请先 pip install ultralytics==8.4.10")
        return 1

    root = Path(ultralytics.__file__).parent
    print(f"ultralytics 位置: {root}")
    print(f"ultralytics 版本: {ultralytics.__version__}")
    if ultralytics.__version__ != EXPECTED_VERSION:
        print(f"[警告] 本补丁基于 {EXPECTED_VERSION} 验证，当前版本为 {ultralytics.__version__}，")
        print("       若锚点匹配失败，请参考 README 第 6 节手动修改。")

    if args.restore:
        restore(root)
        return 0

    print("\n[1/3] 处理 block.py")
    if not patch_block(root):
        return 1
    print("\n[2/3] 处理 tasks.py 与 __init__.py")
    for rel in EDITS:
        if not apply_edits(rel, root):
            return 1

    print("\n[3/3] 完成")
    if not args.no_build:
        ok = build_check(Path(args.yaml))
        if not ok:
            return 1
    print("\n全部完成。现在可以按 README 第 7 节开始训练：")
    print("yolo detect train model=models/yolo26-p2-repncsp.yaml data=<你的VisDrone.yaml> ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
