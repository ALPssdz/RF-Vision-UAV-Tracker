#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_and_train_yolo.py — RF-Vision 5.8GHz YOLO 数据集生成 + 训练一键脚本
===========================================================================

标注策略（关键）：
  RFUAV 数据集为纯无人机录制，每个物理窗口均含无人机信号，
  无需 S3 伪标注 —— 直接全部标注为正样本（UAV_Signal）。
  负样本由合成 AWGN（加性高斯白噪声）生成，功率与录制噪底一致。

数据流：
  Drone RF Data/*.iq (fp32 复数, 100 MSps, USRP X310)
       ↓ scipy.resample_poly(2, 5)  —— 无混叠多相滤波
  40 MSps 复数时域信号（与 AD9364 采样率完全一致）
       ↓ 切分为 2,621,440 样本窗口（65.5 ms）
       ↓ 功率检验：mean(|x|²) > PWR_GATE_FP32 = 1e-9（过滤损坏帧）
       ↓ 与 rf_stage2_waterfall_yolo.py 完全相同的 STFT 管线
  640×640 VIRIDIS 瀑布图（FFT=2048, Blackman, 均值池化, vmin=-63, vmax=27）
       ↓ 全部标注为 UAV_Signal + 合成负样本
  YOLO 格式数据集 → YOLOv8n 训练
  rf_zynq/yolo/best.pt

色彩映射：cv2.COLORMAP_VIRIDIS（蓝→绿→黄）
  ⚠ 旧数据集使用 COLORMAP_HOT，与当前系统不符，已废弃并强制清空。

用法：
  python tools/build_and_train_yolo.py              # 完整流程
  python tools/build_and_train_yolo.py --skip-gen   # 跳过生成，直接训练
  python tools/build_and_train_yolo.py --skip-train # 仅生成数据集
"""

import os, sys, re, shutil, random, time, logging, argparse
from pathlib import Path

# ── 项目根目录加入搜索路径 ──────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# 配置常量（必须与 rf_stage2_waterfall_yolo.py 保持完全一致）
# ============================================================
FFT_SIZE     = 2048
HOP_SIZE     = FFT_SIZE // 2          # 1024，50% 重叠
TARGET_W     = 640
TARGET_H     = 640
VMIN         = -63.0
VMAX         = 27.0
POOL_SIZE    = FFT_SIZE // TARGET_W   # = 3
BLACKMAN_WIN = np.blackman(FFT_SIZE).astype(np.float32)
# !! VIRIDIS = 当前系统色彩。HOT 为旧版训练色彩，已弃用。
COLORMAP     = cv2.COLORMAP_VIRIDIS

# ── 采样率参数 ──────────────────────────────────────────────
FS_SRC    = 100_000_000   # RFUAV 录制采样率（100 MSps）
FS_DST    =  40_000_000   # 目标采样率（40 MSps，AD9364）
RESAMP_UP = 2
RESAMP_DN = 5

# ── 窗口大小（等同于 S2 DMA 缓冲区） ──────────────────────────
WINDOW_SIZE = 2_621_440

# ── 路径 ──────────────────────────────────────────────────
IQ_ROOT   = PROJ_ROOT / "Drone RF Data"
DS_ROOT   = PROJ_ROOT / "rf_yolo_dataset"
YAML_PATH = DS_ROOT / "rf_uav.yaml"
BEST_DST  = PROJ_ROOT / "rf_zynq" / "yolo" / "best.pt"

# ── 训练超参 ──────────────────────────────────────────────
EPOCHS       = 100
BATCH        = 16
IMG_SIZE     = 640
VAL_RATIO    = 0.20
NEG_POS_RATE = 1.0    # 负:正 = 1:1

# ── fp32 数据功率门限（USRP fp32 量程，极低阈值，仅过滤损坏帧） ──
# RFUAV 信号功率典型范围：1e-6 ~ 1e-2（fp32 归一化）
# 此处设为 1e-9，仅排除全零/损坏段，不过滤正常信号
PWR_GATE_FP32 = 1e-9

# ── 合成负样本：AWGN 噪声标准差（匹配真实底噪量级） ──
# 典型 USRP 底噪功率约 1e-5，std ≈ sqrt(1e-5) ≈ 3.16e-3
AWGN_STD = 3e-3


# ============================================================
# 工具函数
# ============================================================

def read_iq_fp32(path: Path) -> np.ndarray:
    """读取 RFUAV fp32 IQ 文件 → complex64。
    格式：[I0 Q0 I1 Q1 ...] float32 小端序。"""
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 2:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def resample_to_40m(x: np.ndarray) -> np.ndarray:
    """100 MSps → 40 MSps 多相滤波（无混叠）。实部虚部分别处理。"""
    from scipy.signal import resample_poly
    r = resample_poly(x.real, RESAMP_UP, RESAMP_DN).astype(np.float32)
    i = resample_poly(x.imag, RESAMP_UP, RESAMP_DN).astype(np.float32)
    return (r + 1j * i).astype(np.complex64)


def make_waterfall(iq_win: np.ndarray) -> np.ndarray:
    """
    640×640 VIRIDIS 瀑布图 —— 与 rf_stage2_waterfall_yolo.py 实现完全相同。

    输入：complex64, 2621440 样本, fp32 归一化（约 ±1.0 范围）
    输出：640×640×3 BGR uint8, VIRIDIS colormap
    """
    iq = iq_win.astype(np.complex64)
    iq -= iq.mean()                                    # DC 去除

    # 向量化 STFT（等间隔取 640 帧）
    N = len(iq)
    n_total = (N - FFT_SIZE) // HOP_SIZE + 1
    idx     = np.linspace(0, n_total - 1, TARGET_H, dtype=int)
    starts  = idx * HOP_SIZE
    frames  = np.array([iq[s: s + FFT_SIZE] for s in starts], dtype=np.complex64)

    windowed = frames * BLACKMAN_WIN
    fft_out  = np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)
    power_db = 20.0 * np.log10(np.abs(fft_out).astype(np.float32) + 1e-12)

    # 线性功率均值池化（3 bin → 1 像素）
    trim = (FFT_SIZE - POOL_SIZE * TARGET_W) // 2
    trimmed   = power_db[:, trim: FFT_SIZE - trim]
    lin       = 10.0 ** (trimmed / 10.0)
    pooled    = np.mean(lin.reshape(TARGET_H, TARGET_W, POOL_SIZE), axis=2)
    wf        = 10.0 * np.log10(pooled + 1e-20)

    # 归一化 + VIRIDIS 映射
    wf_norm = ((np.clip(wf, VMIN, VMAX) - VMIN) / (VMAX - VMIN) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(wf_norm, COLORMAP)


def check_power(iq_win: np.ndarray) -> bool:
    """简单功率检验：过滤损坏/全零帧。
    RFUAV fp32 数据功率典型范围 1e-6~1e-2，阈值设极低（1e-9）。
    """
    return float(np.mean(np.abs(iq_win) ** 2)) > PWR_GATE_FP32


def make_awgn_window() -> np.ndarray:
    """生成合成 AWGN 负样本窗口（fp32 复数，功率匹配底噪）。"""
    real = np.random.normal(0, AWGN_STD, WINDOW_SIZE).astype(np.float32)
    imag = np.random.normal(0, AWGN_STD, WINDOW_SIZE).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def yolo_label(vtsbw_mhz: int, fs_mhz: float = 40.0) -> str:
    """
    生成 YOLO txt 标签（归一化 xywh）。

    频谱图 x 轴: [-20 MHz, +20 MHz]，中心像素 = 0 Hz。
    OcuSync 视频信号占据中心 ±(vtsbw/2) MHz。

    归一化宽度 = vtsbw_mhz / fs_mhz
    """
    w = min(vtsbw_mhz / fs_mhz, 1.0)
    return f"0 0.500000 0.500000 {w:.6f} 0.900000"


def augment(img: np.ndarray, n: int = 2) -> list:
    """轻量频谱图增强：随机亮度/对比度扰动（不翻转，频率/时间方向有物理含义）。"""
    out = []
    for _ in range(n):
        alpha = 1.0 + random.uniform(-0.08, 0.08)
        beta  = random.randint(-5, 5)
        aug   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        out.append(aug)
    return out


def write_yaml():
    """写入（或更新）rf_uav.yaml，路径使用绝对路径。"""
    content = (
        f"train: {DS_ROOT / 'images' / 'train'}\n"
        f"val:   {DS_ROOT / 'images' / 'val'}\n\n"
        f"nc: 1\n"
        f"names: ['UAV_Signal']\n"
    )
    YAML_PATH.write_text(content)
    log.info(f"已更新 YAML: {YAML_PATH}")


# ============================================================
# Phase 1: 数据集生成
# ============================================================

def phase1_generate(clean: bool = True):
    log.info("=" * 64)
    log.info("Phase 1  IQ → VIRIDIS 瀑布图 → YOLO 数据集")
    log.info("标注策略：RFUAV 全为无人机数据 → 直接全标为正样本")
    log.info("负样本：合成 AWGN（功率匹配录制底噪）")
    log.info("=" * 64)

    # 目录
    for split in ("train", "val"):
        (DS_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DS_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    if clean:
        log.info("清空旧数据集（COLORMAP_HOT → COLORMAP_VIRIDIS，不兼容）...")
        for split in ("train", "val"):
            for p in (DS_ROOT / "images" / split).glob("*.jpg"):
                p.unlink()
            for p in (DS_ROOT / "labels" / split).glob("*.txt"):
                p.unlink()

    write_yaml()

    # 扫描全部 .iq 文件
    jobs = []   # (iq_path, vtsbw_mhz, drone_tag)
    for drone_dir in sorted(IQ_ROOT.iterdir()):
        if not drone_dir.is_dir():
            continue
        tag = drone_dir.name.replace(" ", "_")
        for vtsbw_dir in sorted(drone_dir.iterdir()):
            if not vtsbw_dir.is_dir():
                continue
            m = re.search(r"(\d+)", vtsbw_dir.name)
            vtsbw = int(m.group(1)) if m else 10
            for iq_f in sorted(vtsbw_dir.glob("*.iq")):
                jobs.append((iq_f, vtsbw, tag))

    log.info(f"找到 {len(jobs)} 个 .iq 文件（全部含无人机信号）\n")

    pos_bank = []   # (img, label_str, stem)
    skipped  = 0

    for ji, (iq_path, vtsbw, tag) in enumerate(jobs):
        log.info(f"[{ji+1}/{len(jobs)}] {tag}  VTSBW={vtsbw}MHz  {iq_path.name}")

        try:
            iq_raw = read_iq_fp32(iq_path)
        except Exception as e:
            log.warning(f"  跳过（读取失败）: {e}")
            continue

        iq_40m = resample_to_40m(iq_raw)
        del iq_raw

        n_win = len(iq_40m) // WINDOW_SIZE
        log.info(f"  降采样完成，共 {n_win} 个窗口 → 全部标注为正样本")

        pwr_list = []
        for wi in range(n_win):
            win  = iq_40m[wi * WINDOW_SIZE: (wi + 1) * WINDOW_SIZE]

            # 仅过滤损坏帧（全零/NaN）
            if not check_power(win):
                log.warning(f"  [!] w{wi:03d}: 功率过低（损坏帧），跳过")
                skipped += 1
                continue

            pwr = float(np.mean(np.abs(win) ** 2))
            pwr_list.append(pwr)

            stem = f"{tag}_vtsbw{vtsbw}_{iq_path.stem}_w{wi:03d}"
            bgr  = make_waterfall(win)
            pos_bank.append((bgr, yolo_label(vtsbw), stem))

        if pwr_list:
            log.info(
                f"  功率统计: min={min(pwr_list):.2e}  "
                f"max={max(pwr_list):.2e}  "
                f"mean={sum(pwr_list)/len(pwr_list):.2e}  "
                f"(全 {len(pwr_list)} 窗口标为正)"
            )
        del iq_40m

    log.info(f"\n原始正样本: {len(pos_bank)}  损坏跳过: {skipped}")

    # 增强正样本 ×2
    log.info("正样本增强（×2 亮度扰动）...")
    aug_pos = []
    for img, lbl, stem in pos_bank:
        for ai, aug_img in enumerate(augment(img, 2)):
            aug_pos.append((aug_img, lbl, f"{stem}_aug{ai}"))
    all_pos = pos_bank + aug_pos
    log.info(f"增强后正样本: {len(all_pos)}")

    # ── 合成 AWGN 负样本 ──────────────────────────────────────────
    n_neg = int(len(all_pos) * NEG_POS_RATE)
    log.info(f"生成 {n_neg} 个合成 AWGN 负样本（std={AWGN_STD}）...")
    neg_bank = []
    for ni_idx in range(n_neg):
        awgn_win = make_awgn_window()
        bgr      = make_waterfall(awgn_win)
        neg_bank.append((bgr, f"neg_awgn_{ni_idx:05d}"))

    log.info(f"负样本生成完成：{len(neg_bank)} 张")

    # ── 分割 train / val ──────────────────────────────────────────
    random.shuffle(all_pos)
    random.shuffle(neg_bank)
    n_vp = max(1, int(len(all_pos) * VAL_RATIO))
    n_vn = max(1, int(len(neg_bank) * VAL_RATIO))

    def save_pos(samples, split):
        for img, lbl, stem in samples:
            cv2.imwrite(
                str(DS_ROOT / "images" / split / f"{stem}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            (DS_ROOT / "labels" / split / f"{stem}.txt").write_text(lbl)

    def save_neg(samples, split):
        for img, stem in samples:
            cv2.imwrite(
                str(DS_ROOT / "images" / split / f"{stem}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            # 负样本：不写 .txt → YOLO 视为背景

    save_pos(all_pos[:n_vp],  "val")
    save_pos(all_pos[n_vp:],  "train")
    save_neg(neg_bank[:n_vn], "val")
    save_neg(neg_bank[n_vn:], "train")

    for split in ("train", "val"):
        ni = len(list((DS_ROOT / "images" / split).glob("*.jpg")))
        nl = len(list((DS_ROOT / "labels" / split).glob("*.txt")))
        log.info(f"  {split}: {ni} 张图像，{nl} 个正标签，{ni-nl} 张负样本")

    log.info("Phase 1 完成\n")


# ============================================================
# Phase 2: YOLOv8 训练
# ============================================================

def phase2_train():
    log.info("=" * 64)
    log.info("Phase 2  YOLOv8n 训练（5.8GHz OcuSync 瀑布图检测）")
    log.info("=" * 64)

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("缺少 ultralytics，请先: pip install ultralytics")
        sys.exit(1)

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    log.info(f"训练设备: {'CUDA GPU[0]' if device == '0' else 'CPU（较慢）'}")

    if not YAML_PATH.exists():
        log.error(f"YAML 不存在: {YAML_PATH}，请先运行 Phase 1")
        sys.exit(1)

    # 优先迁移学习
    if BEST_DST.exists():
        log.info(f"发现已有权重，迁移学习: {BEST_DST}")
        model = YOLO(str(BEST_DST))
    else:
        log.info("使用 YOLOv8n 预训练权重（ImageNet backbone）")
        # 搜索本地 yolov8n.pt（优先使用项目内置）
        candidate_dirs = [
            PROJ_ROOT / "rf_zynq" / "yolo",                         # ← 项目内置（优先）
            PROJ_ROOT,
            Path.cwd(),
            Path.home() / "AppData" / "Roaming" / "Ultralytics",   # Windows 缓存
            Path.home() / ".config" / "Ultralytics",                # Linux 缓存
        ]
        found_pt = None
        for d in candidate_dirs:
            p = d / "yolov8n.pt"
            if p.exists():
                found_pt = p
                break
        model = YOLO(str(found_pt) if found_pt else "yolov8n.pt")

    run_dir = PROJ_ROOT / "rf_zynq" / "yolo" / "runs"

    results = model.train(
        data       = str(YAML_PATH),
        epochs     = EPOCHS,
        batch      = BATCH,
        imgsz      = IMG_SIZE,
        device     = device,
        project    = str(run_dir),
        name       = "train",
        exist_ok   = True,
        verbose    = True,
        # ─── 频谱图专用增强（禁用空间变换，保留物理意义） ───────────
        hsv_h      = 0.0,    # 色调不变（VIRIDIS 色彩有物理意义）
        hsv_s      = 0.0,    # 饱和度不变
        hsv_v      = 0.05,   # 亮度微扰
        flipud     = 0.0,    # 禁止上下翻转（时间轴有方向）
        fliplr     = 0.0,    # 禁止左右翻转（频率轴有方向）
        mosaic     = 0.3,    # 轻度马赛克增强
        mixup      = 0.0,    # 禁止 mixup（会混淆频谱特征）
        copy_paste = 0.0,
    )

    # 复制最优权重
    best_run = run_dir / "train" / "weights" / "best.pt"
    if best_run.exists():
        BEST_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_run), str(BEST_DST))
        log.info(f"最优权重已保存: {BEST_DST}")
    else:
        log.warning("未找到 best.pt，请检查训练输出目录")

    try:
        m = results.results_dict
        log.info("\n======= 训练结果 =====================")
        log.info(f"  mAP@0.5        = {m.get('metrics/mAP50(B)', 0):.4f}")
        log.info(f"  mAP@0.5:0.95   = {m.get('metrics/mAP50-95(B)', 0):.4f}")
        log.info(f"  Precision      = {m.get('metrics/precision(B)', 0):.4f}")
        log.info(f"  Recall         = {m.get('metrics/recall(B)', 0):.4f}")
        log.info("======================================\n")
    except Exception:
        pass

    log.info("Phase 2 完成")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF-Vision YOLO 数据集生成 + 训练一键脚本"
    )
    parser.add_argument(
        "--skip-gen", action="store_true",
        help="跳过数据集生成，直接进入训练（需已存在 rf_yolo_dataset/）"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="仅生成数据集，不执行训练"
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="不清空旧数据集（增量追加模式）"
    )
    args = parser.parse_args()

    t0 = time.time()

    if not args.skip_gen:
        phase1_generate(clean=not args.no_clean)

    if not args.skip_train:
        phase2_train()

    log.info(f"全流程完成，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
