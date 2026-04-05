#!/bin/bash
# start_rf_vision.sh — RF-Vision-UAV-Tracker 一键启动脚本
# 将此文件放在项目根目录，双击桌面快捷方式即可运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  RF-Vision-UAV-Tracker 启动中..."
echo "  工作目录: $SCRIPT_DIR"
echo "========================================"

python3 system_hub.py

# 程序退出后暂停，让用户看到错误信息
echo ""
echo "========================================="
echo "  程序已退出。按 Enter 关闭窗口。"
echo "========================================="
read
