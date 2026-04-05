# RF-Vision-UAV-Tracker

[English](README.md) | **简体中文**

## 目录
- [1. 系统引言](#1-系统引言)
- [2. 系统硬件架构](#2-系统硬件架构)
- [3. 三级级联射频检测流水线](#3-三级级联射频检测流水线)
- [4. 非对称融合体制设计](#4-非对称融合体制设计)
- [5. 软件栈与模块组织](#5-软件栈与模块组织)
- [6. 部署与装配指南](#6-部署与装配指南)

## 1. 系统引言
RF-Vision-UAV-Tracker 是一套分布式多模态无人机（UAV）探测与预警系统。本系统通过聚合软件无线电（SDR）技术与边缘计算光学视觉技术，规避了传统单一传感器方案的固有物理局限（如天顶极化盲区与无线电静默欺骗）。系统底层采用非对称带外（Out-Of-Band, OOB）传感器融合机制，在复杂电磁环境下实现高鲁棒性的目标截获与多模态取证记录。

中央控制节点运行于 **香橙派 5（RK3588）** 平台，通过 RKNN-Toolkit2 调用芯片内置的 **NPU（神经网络处理单元）**，对射频频谱瀑布图执行硬件加速 YOLOv8 推理，单帧延迟约 20–40 ms，较纯 CPU 推理提升约 **10 倍**。

## 2. 系统硬件架构
系统硬件拓扑基于千兆以太网局域网（LAN）构建，连接三个高度解耦的物理计算节点：

*   **射频传感探测节点（ZYNQ-7020 + AD9364）**
    主干全向探测阵列。利用 AD9364 收发器 56 MHz 大瞬时调谐带宽，配接垂直极化双频天线，对 5.8 GHz ISM 频段（DJI OcuSync 专用信道）执行连贯频谱测绘与特征跳频截获。通过 `libiio` / `pyadi-iio` 协议将 IQ 码流经局域网 TCP/IP 透传至中央控制节点。

*   **视觉光电传感节点（Kendryte K230）**
    天顶补偿节点。挂载 1080P 光学传感器，使用内部 KPU 执行硬件级 YOLO 推理，弥补全向射频天线固有的"天顶极化零陷（Zenith Null）"盲区。视频数据经 RTSP 高带宽链路传送，目标锁定结论（边界框坐标 + 置信度）通过 UDP 低延迟带外信令独立播发。

*   **主控调度大核（香橙派 5 — RK3588）**
    全局事件总线与聚合处理枢纽。执行三级射频检测流水线，通过 RKNN-Toolkit-Lite2 在 RK3588 NPU 上运行 YOLOv8 瀑布图推理，完成多模态证据融合，驱动 PyQt5 上位机界面实时可视化并将告警事件持久化至 SQLite3 数据库。

## 3. 三级级联射频检测流水线

```
IQ 码流采集（AD9364，采样率 40 MSps，单次捕获 262 万采样点 / 65 ms）
        │
        ▼
  第一级 — RSSI 快速功率预扫（S1）
    宽带功率测量，定位 5.8 GHz 各信道信号强度排名，
    以最小时间开销确定优先驻留扇区。
        │
        ▼
  第二级 — 频谱成像 + YOLOv8 推理（S2）
    短时傅里叶变换（STFT）瀑布图（640×640，HOT 伪彩色映射）。
    YOLOv8n 在 RK3588 NPU 上运行 RKNN INT8 推理（约 30 ms/帧）。
        │
        ▼
  第三级 — 循环谱物理层审计（S3）
    基于 OFDM 循环前缀（CP）时移自相关进行协议指纹识别。
    OcuSync 2.0（τ=2667，Δf=15 kHz）与 OcuSync 3.0/4.0（τ=1333，Δf=30 kHz）。
    独立告警判决；Wi-Fi 自适应虚警抑制机制。
```

## 4. 非对称融合体制设计
有别于常规布尔 AND 逻辑融合（须两路全通），本系统实施独立异步越位触发机制，最大化预警召回率：

1.  **射频触发（第一类触发源）** — S3 循环谱物理层审计确认 OcuSync 协议指纹后独立触发告警，并生成频谱取证快照。
2.  **视觉信令触发（第二类触发源）** — K230 UDP 遥测独立触发告警，补偿无线电静默或穿越天线零陷的目标。

两类触发路径均生成融合证据复合图（射频瀑布图 + 光学帧），存入 SQLite3 告警数据库。

## 5. 软件栈与模块组织

```
RF-Vision-UAV-Tracker/
├── system_hub.py            # 系统入口与中央管线调度引擎
├── config.py                # 硬件参数统一配置中心
├── backend_rk3588/
│   └── main_rf_pipeline.py  # RFToolchain：S1→S2→S3 流水线主控
├── rf_zynq/
│   ├── rf_stage1_rssi_scan.py       # S1：跨扇区 RSSI 快速功率预扫
│   ├── rf_stage2_waterfall_yolo.py  # S2：IQ → STFT 瀑布图张量生成
│   ├── rf_stage3_cyclostationary.py # S3：循环谱审计与协议指纹识别
│   └── rknn_infer.py                # RKNN-Lite2 YOLOv8 NPU 推理封装
├── vision_k230/
│   └── k230_client.py       # RTSP 视频流 + UDP 遥测并发网络客户端
├── ui_qt/
│   └── gui_host.py          # PyQt5 纯表现层（View 组件，禁止干涉业务逻辑）
├── database/
│   └── db_manager.py        # SQLite3 告警持久化与 LRU 容量管理
├── tools/
│   └── convert_yolo_to_rknn.py  # YOLOv8 → RKNN INT8 离线转换工具
├── mock_transmitter/
│   └── mock_k230.py         # PC 侧 K230 模拟器（MJPEG 流 + UDP 遥测）
├── deploy_orangepi.sh       # 香橙派 5 一键环境装配脚本
└── start_rf_vision.sh       # 系统一键拉起脚本
```

## 6. 部署与装配指南

### 6.1 香橙派 5（RK3588）—— 生产运行环境
```bash
# 克隆仓库并执行自动化环境配置
git clone https://github.com/ALPssdz/RF-Vision-UAV-Tracker.git
cd RF-Vision-UAV-Tracker
bash deploy_orangepi.sh

# 在 x86 Linux / WSL2 上将 YOLOv8 权重转换为 RKNN INT8 模型
python tools/convert_yolo_to_rknn.py

# 将 best.rknn 复制至目标路径后启动系统
python3 system_hub.py
```

### 6.2 Windows / x86 Linux —— 开发调试模式
```bash
pip install pyadi-iio PyQt5 opencv-python numpy torch ultralytics
python system_hub.py
```
> 若未找到 RKNN 模型（`best.rknn`），系统自动降级为 PyTorch CPU 推理并输出警告，S1/S3 射频检测全链路不受影响。

### 6.3 K230 视觉节点模拟器（PC 侧）
```bash
# 在 PC 上启动 Mock K230 服务器，并将 config.py 中 K230_RTSP_URL 指向该地址
python mock_transmitter/mock_k230.py
```
