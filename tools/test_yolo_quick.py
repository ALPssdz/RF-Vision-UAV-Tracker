#!/usr/bin/env python3
"""快速诊断：验证训练后的 YOLO 模型是否能正确检测训练集中的瀑布图"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

# ── 1. 查找模型 ──
runs_root = PROJ / "rf_zynq" / "yolo" / "runs"
candidates = list(runs_root.glob("*/weights/best.pt"))
if not candidates:
    print("[ERROR] 未找到 best.pt")
    sys.exit(1)
best_pt = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
print(f"[1] 加载模型: {best_pt}")
model = YOLO(str(best_pt))

# ── 2. 取训练集中前 5 张正样本测试 ──
ds = PROJ / "rf_yolo_dataset"
train_imgs = sorted((ds / "images" / "train").glob("*.jpg"))
train_lbls = ds / "labels" / "train"

# 筛选有标签的（正样本）
pos_imgs = [p for p in train_imgs if (train_lbls / (p.stem + ".txt")).exists()]
print(f"[2] 训练集正样本: {len(pos_imgs)} 张，取前5张测试推理\n")

for img_path in pos_imgs[:5]:
    img = cv2.imread(str(img_path))
    results = model.predict(source=img, verbose=False, conf=0.25)
    
    n_boxes = 0
    max_conf = 0.0
    if results:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                n_boxes = len(r.boxes)
                max_conf = float(np.max(confs))
    
    status = "✓ DETECTED" if max_conf > 0.5 else "✗ MISSED"
    print(f"  {status}  conf={max_conf:.4f}  boxes={n_boxes}  {img_path.name}")

# ── 3. 取1张负样本测试 ──
neg_imgs = [p for p in train_imgs if not (train_lbls / (p.stem + ".txt")).exists()]
if neg_imgs:
    img = cv2.imread(str(neg_imgs[0]))
    results = model.predict(source=img, verbose=False, conf=0.25)
    max_conf = 0.0
    if results:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                max_conf = float(np.max(confs))
    print(f"\n  负样本: conf={max_conf:.4f}  {neg_imgs[0].name}")
    print(f"  {'✓ 正确拒绝' if max_conf < 0.5 else '✗ 误检！'}")

print("\n[完成] 如果训练集推理正常但实际运行为0，则为域偏移问题。")
