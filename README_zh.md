# RF-Vision-UAV-Tracker

[English](README.md) | **简体中文**

![平台](https://img.shields.io/badge/平台-Orange%20Pi%205%20%7C%20RK3588-orange)
![SDR](https://img.shields.io/badge/SDR-ZYNQ--7020%20%2B%20AD9364-blue)
![视觉](https://img.shields.io/badge/视觉节点-Kendryte%20K230-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

分布式多模态无人机（UAV）探测与预警系统，融合软件无线电（SDR）与边缘计算光学视觉技术，通过非对称带外（OOB）传感器融合架构，克服单一传感器的天顶极化盲区与无线电静默欺骗问题，实现复杂电磁环境下的鲁棒目标截获与多模态取证记录。

---

## 目录

- [系统硬件架构](#系统硬件架构)
- [三级级联射频检测流水线](#三级级联射频检测流水线)
- [S3 自主现场校准机制](#s3-自主现场校准机制)
- [非对称传感器融合策略](#非对称传感器融合策略)
- [软件模块组织](#软件模块组织)
- [硬件与软件前提条件](#硬件与软件前提条件)
- [部署与配置指南](#部署与配置指南)
- [实测检测性能](#实测检测性能)

---

## 系统硬件架构

三个高度解耦的物理计算节点经千兆以太网局域网互联：

```
┌──────────────────────────────────┐    TCP/IP（IQ 码流）    ┌──────────────────────────────────────────┐
│   射频传感探测节点               │ ──────────────────────► │   主控调度大核                           │
│   ZYNQ-7020 + AD9364             │                         │   香橙派 5（RK3588）                     │
│   • 56 MHz 瞬时调谐带宽          │                         │   • 三级级联射频检测流水线（v4.0）        │
│   • 5.8 GHz 垂直极化双频天线     │                         │   • YOLOv8 NPU 推理（RKNN FP16）         │
│   • libiio / pyadi-iio           │                         │   • 多模态证据融合与告警持久化            │
└──────────────────────────────────┘                         │   • PyQt5 上位机界面 + SQLite3 数据库    │
                                                             └──────────────────────────────────────────┘
┌──────────────────────────────────┐   RTSP 视频流 + UDP 告警遥测          ▲
│   视觉光电传感节点               │ ─────────────────────────────────────►│
│   Kendryte K230                  │
│   • 1080P 光学传感器             │
│   • 内置 KPU 运行 YOLO 推理      │
│   • 补偿天线天顶极化盲区         │
└──────────────────────────────────┘
```

### 节点职能

| 节点 | 硬件 | 职能 |
|------|------|------|
| **射频传感节点** | ZYNQ-7020 + AD9364 | 5.8 GHz 全向 IQ 码流实时采集 |
| **视觉光电节点** | Kendryte K230 | 天顶盲区补偿，光学 YOLO 目标识别 |
| **主控调度大核** | 香橙派 5（RK3588） | 流水线调度、证据融合、GUI 与告警数据库管理 |

---

## 三级级联射频检测流水线

```
IQ 码流采集（AD9364，采样率 40 MSps，单次捕获 262 万采样点，约 65 ms）
        │
        ▼ 第一级 — 峰度加权 RSSI 快速预扫（S1）
        │
        ▼ 第二级 — 频谱成像 + YOLOv8 NPU 推理（S2）
        │
        ▼ 第三级 — 循环频率判别器（S3）
        │
        ▼ TPF — 三级弹性时序确认滤波器
        │
        ▼ 告警事件 → SQLite3 数据库 + GUI 可视化
```

### 第一级 — 峰度加权 RSSI 快速预扫（S1）

跨扇区宽带功率测量，引入突发帧峰度加权排名，优先调度弱信号扇区进行深度分析。

**峰度加权扇区评分公式：**

$$\tilde{P}_f = \bar{P}_f \cdot \left(1 + \beta \cdot \frac{\kappa_f - 3}{3}\right), \quad \beta = 0.40$$

OcuSync 低占空比突发帧（κ ≈ 6~8）的扇区评分比热噪声（κ ≈ 3）高出 **40~80%**，确保弱信号扇区被优先选入深度分析。

| 参数 | 值 |
|------|----|
| 缓冲区大小 | 524,288 采样点（13.1 ms） |
| 时序滤波 | 3 帧中值滤波 |

### 第二级 — 频谱成像 + YOLOv8 NPU 推理（S2）

将 IQ 码流转换为二维时频瀑布图，在 RK3588 NPU 上进行硬件加速推理。

| 项目 | 参数 |
|------|------|
| 变换方法 | 短时傅里叶变换（STFT），Blackman 窗 → 640×640 VIRIDIS 伪彩色瀑布图 |
| 模型 | YOLOv8n，RKNN FP16 在 RK3588 NPU 上运行 |
| 推理延迟 | 约 30 ms / 帧 |
| 训练数据集 | RFUAV IQ 数据集（131 段录制，5 款无人机，重采样至 40 MSps）+ 合成 AWGN 负样本 |
| 训练 mAP@0.5 | **0.995** |
| 实际 SDR 置信度 | ≈ 0.2~0.7（域偏移，属正常现象） |

S2 的 `bbox_score` 输出透传至第三级，用于软判决评分融合（SDS）注入。

### 第三级 — 循环频率判别器（S3）

四重正交互验证（TOV）+ 软判决评分融合（SDS）双引擎驱动。

#### CAF-FFT 算法核心

```
z[n] = x[n] · x*[n−τ]
NCC[α] = |FFT(z)[k]| / (N_z · P_x)
```

**Wi-Fi 对 OcuSync 通道的理论泄漏**（N=160,000，Fs=40 MHz）：

```
NCC_WiFi ≈ P_WiFi × sinc(1130) ≈ P_WiFi 的 0.028%   ← 完全正交隔离
```

**各协议目标循环频率**（Fs = 40 MSps）：

| 协议 | 子载波间隔 Δf | 延迟 τ | 循环频率 α_sym |
|------|-------------|--------|----------------|
| OcuSync 2.0 | 15 kHz | 2667 | 10.5–14.5 kHz |
| OcuSync 3.0/4.0 | 30 kHz | 1333 | 22–30 kHz |
| Wi-Fi 802.11 | 312.5 kHz | 128 | 250 kHz ← 完全正交 |

#### 软判决评分融合（SDS）

$$S = 0.45 \cdot \frac{\text{NCC}}{th} + 0.25 \cdot \log_{10}\frac{\text{PSR}}{th_\text{PSR}} + 0.20 \cdot \log_{10}\frac{\text{CFS}}{th_\text{CFS}} + 0.10 \cdot \mathbf{1}[\text{AFS 通过}]$$

| 判决条件 | 结论 |
|----------|------|
| S ≥ 1.0 **且** NCC ≥ 0.80 × th | ✅ 检出（软下限救援路径） |
| NCC ≥ 2.5 × th | ✅ 检出（强信号直通旁路） |
| 其余 | ❌ 拒绝 |

#### 四级正交验证层

| 层级 | 描述 | 阈值 |
|------|------|------|
| **L1** | 帧级 CAF-FFT 扫描 | CHUNK=160k，重叠率 80%，峰权重 0.65 |
| **L2** | 联合统计量 vs. 按扇区独立自适应阈值 | — |
| **L3** | τ 域峰值旁瓣比（PSR），剔除 SMPS 纹波 | ≥ 2.2× |
| **L4** | α 域循环频率集中度（CFS），剔除宽带弥散噪声 | ≥ 2.0× |
| **AFS** | α 域帧间频率稳定性 | σ_α < 500 Hz |

> **AFS 设计依据：** OcuSync TCXO 频率漂移 < 500 Hz；SMPS/Wi-Fi 伪峰漂移 ~2~5 kHz → 两者可被干净分离。

#### 三级弹性 TPF 虚警率模型

| 信号强度分支 | TPF 确认次数 N | 最终虚警率 P_fa |
|-------------|--------------|----------------|
| 强信号直通旁路（NCC ≥ 3×th） | 1 | < 0.10% |
| 中等信号（1.8×th ≤ NCC < 3×th） | 2 | ≈ 0.25% |
| 弱信号（th ≤ NCC < 1.8×th） | 3 | ≈ 0.013% |

TPF 采用 streak **衰减机制**替代硬归零：`streak[t+1] = max(0, streak[t] − 0.5)` — 信号短暂中断后 2 tick 内重现，无需重新积累确认计数，降低再截获延迟。

---

## S3 自主现场校准机制

每次 `system_hub.py` 启动时，系统自动执行**零交互环境底噪校准**，校准完成后方进入检测主循环，全程无需任何用户干预。

### 校准算法

`calibrate_s3.py` 针对 5.8 GHz 三个扇区分别采集 N = 8 次 IQ 数据，推导按扇区独立自适应阈值：

```
bg_eff = 0.4 × bg_max + 0.6 × bg_avg          # 抗脉冲加权估计，降低单帧 SMPS 突发干扰权重
th     = max(HARD_FLOOR, bg_eff × NOISE_MARGIN) # 按扇区独立推导，各扇区互不影响
```

### 校准参数说明

| 参数 | 值 | 设计依据 |
|------|----|---------|
| `NOISE_MARGIN` | 2.0× | SDS（PSR+CFS+AFS）承担主要假警报抑制，检测阈值可从紧设置 |
| `BG_MAX_WEIGHT` | 0.40 | 降低单帧 SMPS 突发脉冲对背景估计的统计权重 |
| `HARD_FLOOR_30K` | 1.8% | 理论 NCC 底（1/√160,000 ≈ 0.25%）的 8 倍 |
| `HARD_FLOOR_15K` | 1.4% | 理论 NCC 底的 6 倍 |
| `ALPHA_SCAN_30K` | 22–30 kHz | 覆盖 OcuSync CP=1/8（26.7 kHz）至 CP=1/4（24.0 kHz）全变体 |
| `ALPHA_SCAN_15K` | 10.5–14.5 kHz | 覆盖 OcuSync 15 kHz 信道所有变体 |

校准阈值写入 `rf_zynq/s3_thresholds.json`（已加入 `.gitignore`，自动生成），系统启动时动态加载。**源代码从不被修改**，多机异地部署不产生任何 Git 冲突。

### 校准终端输出示例

```
[Sector 5785MHz]  OcuSync 30kHz: avg=0.31%  max=0.91%
[Sector 5745MHz]  OcuSync 30kHz: avg=0.55%  max=1.21%
[Sector 5825MHz]  OcuSync 30kHz: avg=1.04%  max=2.50%

Per-sector derived thresholds  (NOISE_MARGIN = 2.0×)
5785 MHz:  TH_30k= 1.80%   TH_15k= 1.40%   ← 达到硬底（最净扇区）
5745 MHz:  TH_30k= 2.45%   TH_15k= 1.40%
5825 MHz:  TH_30k= 3.25%   TH_15k= 1.95%
```

---

## 非对称传感器融合策略

有别于常规布尔 AND 逻辑融合（须两路全通才触发），本系统采用**独立异步越位触发机制**，最大化预警召回率：

| 触发类型 | 来源 | 触发条件 |
|----------|------|---------|
| **RF 触发（第一类）** | S3 SDS 判别器 | 四重正交验证确认 OcuSync 协议指纹 |
| **视觉触发（第二类）** | K230 UDP 遥测 | 独立触发，补偿无线电静默或天顶零陷穿越目标 |
| **YOLO 辅助救援** | S2 bbox_score | S3 SDS 得分 ∈ [0.85, 1.00) 时注入 +0.15 补分；**永远不单独触发告警** |

两类主触发路径均生成**融合证据复合图**（射频瀑布图 + 光学帧），持久化存入 SQLite3 告警数据库。

---

## 软件模块组织

```
RF-Vision-UAV-Tracker/
├── system_hub.py                    # 系统入口与中央管线调度引擎（TPF v4.0 三级弹性确认）
│
├── backend_rk3588/
│   ├── config.py                    # 硬件参数统一配置中心 + YOLO_ASSIST_ENABLED 开关
│   └── main_rf_pipeline.py          # RFToolchain：S1→S2→S3 流水线主控
│
├── rf_zynq/
│   ├── rf_stage1_rssi_scan.py       # S1：峰度加权跨扇区 RSSI 快速预扫
│   ├── rf_stage2_waterfall_yolo.py  # S2：IQ → STFT 频谱瀑布图张量生成
│   ├── rf_stage3_cyclostationary.py # S3：CAF-FFT + AFS + SDS 循环频率判别器
│   ├── calibrate_s3.py              # S3 自主现场校准向导
│   ├── s3_thresholds.json           # 运行时阈值（git-ignored，自动生成）
│   └── rknn_infer.py                # RKNN-Lite2 YOLOv8 NPU 推理封装（无 torch 依赖）
│
├── vision_k230/
│   └── k230_client.py               # RTSP 视频流 + UDP 遥测并发网络客户端
│
├── ui_qt/
│   └── gui_host.py                  # PyQt5 纯表现层（View 组件）
│
├── database/
│   └── db_manager.py                # SQLite3 告警持久化与 LRU 容量管理
│
├── tools/
│   ├── build_and_train_yolo.py      # IQ → VIRIDIS 瀑布图数据集生成 + YOLOv8 训练
│   └── convert_yolo_to_rknn.py      # YOLOv8 权重 → RKNN FP16 离线转换工具
│
├── mock_transmitter/
│   ├── uav_tx_gui.py                # PlutoSDR 无人机射频靶机控制台（GUI）
│   │                                  支持机型：DJI Mini 4 Pro / Mavic 3 / Avata 2 / FPV Combo
│   │                                  模式：跳频模式 / 单频模式（可选扇区）
│   └── mock_k230.py                 # PC 侧 K230 模拟器（MJPEG 流 + UDP 遥测）
│
├── deploy_orangepi.sh               # 香橙派 5 首次部署一键环境装配脚本
└── .gitignore
```

---

## 硬件与软件前提条件

### 硬件要求

| 组件 | 规格 |
|------|------|
| 主控节点 | 香橙派 5（RK3588），运行 Ubuntu / Debian |
| 射频节点 | ZYNQ-7020 FPGA 开发板 + AD9364 收发器 |
| 视觉节点 | Kendryte K230 开发板 + 1080P 摄像头 |
| 网络 | 千兆以太网局域网（所有节点同一子网） |

### 软件依赖（主控节点 — 香橙派 5）

```bash
# 核心依赖
pip install pyadi-iio numpy scipy matplotlib Pillow PyQt5

# RKNN 推理运行时（Lite2，设备端无需 PyTorch）
# rknn-toolkit-lite2 通过 deploy_orangepi.sh 预装
```

> **训练 / 转换**（仅限 x86 Linux 或 WSL2 环境）：
> ```bash
> pip install ultralytics rknn-toolkit2
> ```

---

## 部署与配置指南

### 第一步 — 克隆仓库

```bash
git clone https://github.com/ALPssdz/RF-Vision-UAV-Tracker.git
cd RF-Vision-UAV-Tracker
```

### 第二步 — 配置硬件参数

根据实际硬件网络环境修改 `backend_rk3588/config.py`：

```python
SDR_URI             = "ip:192.168.31.10"        # AD9364 在局域网中的 IP 地址
SDR_GAIN_DB         = 70                         # MGC 接收增益（dB），AD9364 最大约 73 dB
SAMPLE_RATE         = int(40e6)                  # ADC 采样率：40 MSps
SWEEP_SECTORS       = [5745e6, 5785e6, 5825e6]  # OcuSync 5.8 GHz 三扇区中心频率（Hz）
YOLO_ASSIST_ENABLED = True                       # 启用 S2 bbox_score → SDS 注入
YOLO_CONF_THRESH    = 0.30                       # 因域偏移适当降低（RFUAV → 实际 SDR）
```

### 第三步 — 启动系统

```bash
# 系统自动执行环境底噪校准（约 60 秒），完成后进入检测主循环
python3 system_hub.py
```

> **注意：** `rf_zynq/s3_thresholds.json` 由各设备本地生成，已排除 Git 追踪。  
> 每当部署地点改变、周围电磁环境显著变化或调整 RX 增益后，需重启 `system_hub.py` 触发重新校准。

---

### 可选 — 构建 YOLO 数据集与重新训练

仓库已附带预训练权重，仅需新增 IQ 录制数据时才需重新训练。

```bash
# 从 RFUAV IQ 录制生成 VIRIDIS 瀑布图数据集并训练 YOLOv8n
python tools/build_and_train_yolo.py

# 若数据集已生成，跳过生成阶段直接训练
python tools/build_and_train_yolo.py --skip-gen
```

### 可选 — 将 YOLOv8 权重转换为 RKNN FP16

需在 **x86 Linux / WSL2** 环境执行（需安装 `rknn-toolkit2`），转换完成后将模型文件复制至香橙派：

```bash
python tools/convert_yolo_to_rknn.py
scp rf_zynq/yolo/best.rknn orangepi@<IP>:/opt/RF-Vision-UAV-Tracker/rf_zynq/yolo/
```

> **为何选用 FP16 而非 INT8？** INT8 量化会显著破坏 YOLOv8 类别置信度评分。  
> FP16 模式（`do_quantization=False`）完整保留推理精度，NPU 推理速度仅比 INT8 慢约 30%。

---

## 实测检测性能

**2026-04-06** 实机验证，使用 PlutoSDR 发射机（DJI Mini 4 Pro / Mavic 3 IQ 数据集）+ AD9364 接收机（70 dB MGC 增益）：

| 事件 | 扇区 | 联合 NCC | 阈值 | PSR | CFS | 结果 |
|------|------|---------|------|-----|-----|------|
| OcuSync 检测 | 5785 MHz | 3.92% | 1.80% | 7.6× | 4.9× | ✅ 已确认 |
| OcuSync 检测（弱信号） | 5785 MHz | 2.02% | 1.80% | 6.0× | 5.8× | ✅ 已确认 |
| OcuSync 检测 | 5785 MHz | 3.46% | 1.80% | 10.1× | 3.5× | ✅ 已确认 |
| OcuSync 检测（强信号） | 5825 MHz | 7.85% | 3.25% | 18.4× | 2.6× | ✅ 已确认 |
| SMPS 突发（α = 14.4 kHz） | 5825 MHz | 9.40% | — | 2.6× | **1.06×** | ❌ 正确剔除（CFS 失败）|
| 宽带噪声 | 5745 MHz | 1.82% | — | **2.35×** | 1.78× | ❌ 正确剔除（PSR 失败）|

**实测循环频率一致性：α = 27.99 kHz** → 对应 OcuSync OFDM CP=1/4 物理结构（N_total = 1429 采样点 @ 40 MSps）。多次检测结果高度一致，证明协议指纹识别稳定可靠。
