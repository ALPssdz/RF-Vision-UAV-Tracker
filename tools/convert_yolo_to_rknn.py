# -*- coding: utf-8 -*-
"""
tools/convert_yolo_to_rknn.py — YOLOv8 → RKNN 模型转换脚本
============================================================
【运行平台】：x86_64 Linux（WSL2 Ubuntu 22.04 / 原生 Linux）
【禁止运行于】：Windows 原生、ARM64（香橙派）

依赖安装：
  pip install rknn-toolkit2 ultralytics onnx

转换流程：
  1. 使用 ultralytics 将 YOLOv8 .pt 导出为 ONNX 格式
  2. 使用 rknn-toolkit2 将 ONNX 量化并转换为 RK3588 专用 .rknn 格式
  3. 输出文件：rf_zynq/yolo/best.rknn（拷贝至香橙派 5 使用）

INT8 量化说明：
  量化需要约 100 张校准图像（频谱瀑布图），从 rf_yolo_dataset/ 目录自动采样。
  量化可将模型体积缩小约 75%（FP32 → INT8），推理速度提升约 2~4 倍。

用法：
  cd <项目根目录>
  python3 tools/convert_yolo_to_rknn.py
"""

import os
import sys
import glob
import platform

# ── 平台检查 ──────────────────────────────────────────────────────────────────
if platform.machine() != 'x86_64':
    print(f"[ERROR] 本脚本须在 x86_64 Linux 上运行，当前架构: {platform.machine()}")
    print(f"        请在 WSL2 (Ubuntu 22.04) 或原生 Linux 上执行。")
    sys.exit(1)

PROJ_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_RKNN  = os.path.join(PROJ_ROOT, "rf_zynq", "yolo", "best.rknn")
OUTPUT_ONNX  = os.path.join(PROJ_ROOT, "rf_zynq", "yolo", "best.onnx")
DATASET_TXT  = os.path.join(PROJ_ROOT, "rf_zynq", "yolo", "quant_dataset.txt")
DATASET_DIR  = os.path.join(PROJ_ROOT, "rf_yolo_dataset")

# ── Step 1：查找最新 YOLOv8 权重 ──────────────────────────────────────────────
print("[Step 1] 查找 YOLOv8 权重文件...")
runs_root = os.path.join(PROJ_ROOT, "rf_zynq", "yolo", "runs")
patterns  = [
    os.path.join(runs_root, "*", "weights", "best.pt"),
    os.path.join(runs_root, "detect", "*", "weights", "best.pt"),
]
matches = []
for p in patterns:
    matches.extend(glob.glob(p))

if not matches:
    print(f"[ERROR] 未找到 best.pt，请确认 YOLO 训练已完成，输出路径: {runs_root}")
    sys.exit(1)

best_pt = sorted(matches, key=os.path.getmtime)[-1]
print(f"[Step 1] 已找到权重: {best_pt}")

# ── Step 2：导出 ONNX ─────────────────────────────────────────────────────────
print("[Step 2] 导出 ONNX 格式...")
from ultralytics import YOLO
model = YOLO(best_pt)
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
# ultralytics 默认将 .onnx 输出至权重同目录
onnx_default = best_pt.replace(".pt", ".onnx")
if os.path.exists(onnx_default):
    import shutil
    shutil.move(onnx_default, OUTPUT_ONNX)
print(f"[Step 2] ONNX 已保存: {OUTPUT_ONNX}")

# ── Step 3：生成量化校准数据集清单 ────────────────────────────────────────────
print("[Step 3] 生成量化校准图像列表...")
img_paths = []
for ext in ["*.jpg", "*.png", "*.jpeg"]:
    img_paths.extend(glob.glob(os.path.join(DATASET_DIR, "**", ext), recursive=True))

# 随机采样至多 200 张（校准集过大不增益准确度，100~200 张足够）
import random
random.shuffle(img_paths)
calib_imgs = img_paths[:200]

if len(calib_imgs) < 20:
    print(f"[WARN] 校准图像不足（仅 {len(calib_imgs)} 张），量化精度可能下降。")
    print(f"       请确认 {DATASET_DIR} 目录中有足够的频谱瀑布图。")

with open(DATASET_TXT, "w") as f:
    for p in calib_imgs:
        f.write(p + "\n")
print(f"[Step 3] 量化校准集: {len(calib_imgs)} 张图像 → {DATASET_TXT}")

# ── Step 4：RKNN 转换与 INT8 量化 ─────────────────────────────────────────────
print("[Step 4] 开始 RKNN 转换（目标平台: rk3588）...")
from rknn.api import RKNN

rknn = RKNN(verbose=False)

# 配置：均值/标准差归一化（YOLOv8 输入为 [0,1]，对应 mean=0, std=255）
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8',   # INT8 量化
    optimization_level=3,
)

ret = rknn.load_onnx(model=OUTPUT_ONNX)
assert ret == 0, f"[ERROR] ONNX 加载失败（code: {ret}）"

ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
assert ret == 0, f"[ERROR] RKNN 编译失败（code: {ret}）"

ret = rknn.export_rknn(OUTPUT_RKNN)
assert ret == 0, f"[ERROR] RKNN 导出失败（code: {ret}）"

rknn.release()
print(f"[Step 4] RKNN 模型已生成: {OUTPUT_RKNN}")
print()
print("=" * 60)
print(f"  转换完成！")
print(f"  将以下文件拷贝至香橙派 5 项目目录下对应位置：")
print(f"    {OUTPUT_RKNN}")
print(f"  拷贝命令示例：")
print(f"    scp rf_zynq/yolo/best.rknn orangepi@<IP>:~/RF-Vision-UAV-Tracker/rf_zynq/yolo/")
print("=" * 60)
