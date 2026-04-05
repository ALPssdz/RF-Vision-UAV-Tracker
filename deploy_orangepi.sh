#!/bin/bash
# deploy_orangepi.sh — 香橙派 5 首次部署脚本
# 用法：bash deploy_orangepi.sh

set -e

echo "=================================="
echo "  RF-Vision-UAV-Tracker 部署脚本"
echo "  目标平台: Orange Pi 5 (RK3588)"
echo "=================================="

# root 不需要 sudo；非 root 自动加 sudo 前缀
if [ "$(id -u)" -eq 0 ]; then
    APT="apt"
else
    APT="sudo apt"
fi

# ── [1/4] 系统依赖 ────────────────────────────────────────────────────────────
echo "[1/4] 安装系统依赖..."
$APT update -qq
$APT install -y python3-pip python3-pyqt5 python3-dev

# ── [2/4] Python 依赖 ─────────────────────────────────────────────────────────
echo "[2/4] 安装 Python 依赖..."
pip3 install --upgrade pip

# 注意：ultralytics >= 8.3.x 依赖 polars，polars 需要 Rust 编译器，
# 在 Python 3.8 / ARM64 上无预编译 wheel，固定至最后兼容版本 8.2.104
pip3 install numpy matplotlib "ultralytics==8.2.103" opencv-python-headless

# ── [3/4] rknn_toolkit_lite2（可选，有 whl 才安装）──────────────────────────
echo "[3/4] 检查 rknn_toolkit_lite2..."
RKNN_WHL=$(ls rf_zynq/yolo/rknn_toolkit_lite2-*.whl 2>/dev/null | head -1)
if [ -n "$RKNN_WHL" ]; then
    pip3 install "$RKNN_WHL"
    echo "      [OK] 已安装 RKNN 推理运行时: $RKNN_WHL"
else
    echo "      [跳过] 未找到 rknn_toolkit_lite2 whl，系统将自动降级为 CPU 推理。"
    echo "      如需 NPU 加速，请从以下地址下载 whl 后重新运行本脚本："
    echo "      https://github.com/airockchip/rknn-toolkit2/releases"
fi

# ── [4/4] 验证 ────────────────────────────────────────────────────────────────
echo "[4/4] 基础验证..."
python3 -c "import numpy, cv2, matplotlib, PyQt5; print('  [OK] 基础依赖验证通过')"
python3 -c "import config; print(f'  [OK] IS_RK3588={config.IS_RK3588} | YOLO_BACKEND={config.YOLO_BACKEND}')"

echo ""
echo "=================================="
echo "  部署完成！"
echo ""
echo "  后续步骤："
echo "  1. 编辑 config.py 确认 SDR_URI 和 K230 地址"
echo "  2. python3 diag_s3_false_positive.py   # 背景噪声验证"
echo "  3. python3 diag_uav_on_calibration.py  # 无人机信号验证"
echo "  4. python3 system_hub.py               # 全系统启动"
echo "=================================="
