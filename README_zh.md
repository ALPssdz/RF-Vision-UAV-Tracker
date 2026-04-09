# RF-Vision-UAV-Tracker

[English](README.md) | **简体中文**

## 目录
- [1. 系统引言](#1-系统引言)
- [2. 系统硬件架构](#2-系统硬件架构)
- [3. 三级级联射频检测流水线 v4.0](#3-三级级联射频检测流水线-v40)
- [4. S3 自主现场校准机制](#4-s3-自主现场校准机制)
- [5. 非对称融合体制设计](#5-非对称融合体制设计)
- [6. 软件栈与模块组织](#6-软件栈与模块组织)
- [7. 部署与装配指南](#7-部署与装配指南)
- [8. 实测检测性能](#8-实测检测性能)

## 1. 系统引言
RF-Vision-UAV-Tracker 是一套分布式多模态无人机（UAV）探测与预警系统。本系统通过聚合软件无线电（SDR）技术与边缘计算光学视觉技术，规避了传统单一传感器方案的固有物理局限（如天顶极化盲区与无线电静默欺骗）。系统底层采用非对称带外（Out-Of-Band, OOB）传感器融合机制，在复杂电磁环境下实现高鲁棒性的目标截获与多模态取证记录。

中央控制节点运行于 **香橙派 5（RK3588）** 平台，通过 RKNN-Toolkit2 调用芯片内置的 **NPU（神经网络处理单元）**，对射频频谱瀑布图执行硬件加速 YOLOv8 推理，显著优于纯 CPU 推理方案的实时性能。

## 2. 系统硬件架构
系统硬件拓扑基于千兆以太网局域网（LAN）构建，连接三个高度解耦的物理计算节点：

*   **射频传感探测节点（ZYNQ-7020 + AD9364）**
    主干全向探测阵列。利用 AD9364 收发器 56 MHz 大瞬时调谐带宽，配接垂直极化双频天线，对 5.8 GHz ISM 频段（DJI OcuSync 专用信道）执行连贯频谱测绘与特征跳频截获。通过 `libiio` / `pyadi-iio` 协议将 IQ 码流经局域网 TCP/IP 透传至中央控制节点。

*   **视觉光电传感节点（Kendryte K230）**
    天顶补偿节点。挂载 1080P 光学传感器，使用内部 KPU 执行硬件级 YOLO 推理，弥补全向射频天线固有的"天顶极化零陷（Zenith Null）"盲区。视频数据经 RTSP 高带宽链路传送，目标锁定结论（边界框坐标 + 置信度）通过 UDP 低延迟带外信令独立播发。

*   **主控调度大核（香橙派 5 — RK3588）**
    全局事件总线与聚合处理枢纽。执行三级射频检测流水线，通过 RKNN-Toolkit-Lite2 在 RK3588 NPU 上运行 YOLOv8 瀑布图推理，完成多模态证据融合，驱动 PyQt5 上位机界面实时可视化并将告警事件持久化至 SQLite3 数据库。

## 3. 三级级联射频检测流水线 v4.0

```
IQ 码流采集（AD9364，采样率 40 MSps，单次捕获 262 万采样点 / 65 ms）
        │
        ▼
  第一级 — 峰度加权 RSSI 快速预扫（S1）  [v4.0 升级]
    宽带功率测量，并引入信号峰度加权排名：
      P̃_f = P̄_f · (1 + β·(κ_f − 3)/3)，β = 0.40
    OcuSync 低占空比突发帧（κ ≈ 6~8，热噪声 κ ≈ 3）的扇区得分
    提升 40~80%，确保弱信号扇区被优先选入深度分析。
    缓冲区：524,288 采样点（13.1 ms），3 帧中值滤波。
        │
        ▼
  第二级 — 频谱成像 + YOLOv8 推理（S2）
    短时傅里叶变换（STFT）瀑布图（640×640，VIRIDIS 伪彩色，Blackman 窗）。
    YOLOv8n 在 RK3588 NPU 上运行 RKNN FP16 推理（约 30 ms/帧）。
    bbox_score 透传至 alert_info，供 SDS 融合使用。
    模型基于 RFUAV IQ 数据集训练（131 段录制，5 款无人机，
    重采样至 40 MSps）+ 合成 AWGN 负样本。
    训练 mAP@0.5 = 0.995；实际 SDR 置信度 ≈ 0.2~0.7（域偏移）。
        │
        ▼
  第三级 — 循环频率判别器 v4.0（S3）
    四重正交互验证（TOV）+ 软判决评分融合（SDS）

    【CAF-FFT 算法核心（继承 v3.x）】
      z[n] = x[n]·x*[n-τ]，NCC[α] = |FFT(z)[k]| / (N_z · P_x)
      WiFi 对 OcuSync 通道的理论泄漏（N=160000, Fs=40 MHz）：
        NCC_WiFi ≈ P_WiFi × sinc(1130) ≈ P_WiFi 的 0.028%

    各协议循环频率（Fs = 40 MSps）：
      OcuSync 2.0（Δf=15 kHz, τ=2667）：α_sym ≈ 10.5–14.5 kHz
      OcuSync 3.0/4.0（Δf=30 kHz, τ=1333）：α_sym ≈ 22–30 kHz
      WiFi 802.11（Δf=312.5 kHz, τ=128）：α_sym = 250 kHz ← 完全正交

    【v4.0 软判决评分融合（SDS）】
      S = 0.45·(NCC/th) + 0.25·log₁₀(PSR/th_PSR)
        + 0.20·log₁₀(CFS/th_CFS) + 0.10·I[AFS 通过]
      S ≥ 1.0 且 NCC ≥ 0.80×th（软下限）→ 检出
      NCC ≥ 2.5×th（强信号）→ 直通旁路

    四级正交验证层：
      L1 — 帧级 CAF-FFT 扫描（CHUNK=160k，重叠率 80%，峰权重 0.65）
      L2 — 联合统计量 > 按扇区独立自适应阈值
      L3 — τ 域峰值旁瓣比（PSR）≥ 2.2×，剔除 SMPS 纹波干扰
      L4 — α 域循环频率集中度（CFS）≥ 2.0×，剔除宽带弥散噪声
      AFS — α 域帧间频率稳定性：σ_α < 500 Hz
            （OcuSync TCXO 漂移 <500 Hz vs. SMPS/WiFi 伪峰漂移 ~2–5 kHz）
```

### 虚警率模型（v4.0）

| 分支 | TPF 确认次数 N | P_fa（最终） |
|------|--------------|------------|
| 强信号直通（NCC ≥ 3×th） | 1 | < 0.10%（AFS+PSR+CFS 保证） |
| 中等信号（1.8×th ≤ NCC < 3×th） | 2 | ≈ 0.25% |
| 弱信号（th ≤ NCC < 1.8×th） | 3 | ≈ 0.013% |

TPF 同时采用 **streak 衰减机制**替代硬归零：
`streak[t+1] = max(0, streak[t] − 0.5)` — 信号短暂中断后 2 tick 内重现
无需重新积累确认计数，降低再截获延迟。

## 4. S3 自主现场校准机制

每次 `system_hub.py` 启动时，系统自动执行**零交互环境底噪校准**，校准完成后方进入检测主循环，无需任何用户干预。

### 校准算法

`calibrate_s3.py` 针对 5.8 GHz 三个扇区分别采集 `N = 8` 次 IQ 数据，计算 CAF-NCC 背景底噪，推导按扇区独立阈值：

```
bg_eff = 0.4 × bg_max + 0.6 × bg_avg          （抗脉冲加权估计，降低 SMPS 突发干扰的权重）
th     = max(HARD_FLOOR, bg_eff × NOISE_MARGIN) （按扇区独立推导，互不影响）
```

| 参数 | 值 | 设计依据 |
|------|----|---------| 
| `NOISE_MARGIN` | 2.0× | SDS 软判决（PSR+CFS+AFS）承担主要假警报抑制，阈值可从紧 |
| `BG_MAX_WEIGHT` | 0.40 | 降低单帧 SMPS 突发脉冲的统计权重 |
| `HARD_FLOOR_30K` | 1.8% | 理论 NCC 底（1/√160000≈0.25%）的 8 倍 |
| `HARD_FLOOR_15K` | 1.4% | 理论 NCC 底的 6 倍 |
| `ALPHA_SCAN_30K` | 22–30 kHz | 覆盖 OcuSync CP=1/8（26.7 kHz）至 CP=1/4（24.0 kHz）全变体 |
| `ALPHA_SCAN_15K` | 10.5–14.5 kHz | 覆盖 OcuSync 15 kHz 信道全变体 |

校准阈值写入 `rf_zynq/s3_thresholds.json`（已加入 `.gitignore`），系统启动时动态加载。**源代码从不被修改**，多机部署不产生 Git 冲突。

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

## 5. 非对称融合体制设计
有别于常规布尔 AND 逻辑融合（须两路全通），本系统实施独立异步越位触发机制，最大化预警召回率：

1.  **射频触发（第一类触发源）** — S3 SDS 判别器通过四重正交验证确认 OcuSync 协议指纹后独立触发告警，生成循环谱取证快照。
2.  **视觉信令触发（第二类触发源）** — K230 UDP 遥测独立触发告警，补偿无线电静默或穿越天线零陷的目标。
3.  **YOLO 辅助救援（已启用）** — 已完成 5.8 GHz RFUAV 瀑布图数据集训练（`config.py` 中 `YOLO_ASSIST_ENABLED = True`）。S2 bbox_score 在 S3 SDS 得分 ∈ [0.85, 1.00) 时注入 +0.15 补充分，对弱信号进行救援检出。YOLO 单独永远不触发告警。

两类主触发路径均生成融合证据复合图（射频瀑布图 + 光学帧），存入 SQLite3 告警数据库。

## 6. 软件栈与模块组织

```
RF-Vision-UAV-Tracker/
├── system_hub.py                    # 系统入口与中央管线调度引擎（TPF v4.0 三级弹性确认）
├── backend_rk3588/
│   ├── config.py                    # 硬件参数统一配置中心 + YOLO_ASSIST_ENABLED 开关
│   └── main_rf_pipeline.py          # RFToolchain：S1→S2→S3 流水线主控（v4.0）
├── rf_zynq/
│   ├── rf_stage1_rssi_scan.py       # S1：峰度加权跨扇区 RSSI 快速预扫（v4.0）
│   ├── rf_stage2_waterfall_yolo.py  # S2：IQ → STFT 瀑布图张量生成
│   ├── rf_stage3_cyclostationary.py # S3：CAF-FFT + AFS + SDS 循环频率判别器（v4.0）
│   ├── calibrate_s3.py              # S3 自主现场校准向导
│   ├── s3_thresholds.json           # 运行时阈值（git-ignored，自动生成）
│   └── rknn_infer.py                # RKNN-Lite2 YOLOv8 NPU 推理封装（无 torch 依赖）
├── vision_k230/
│   └── k230_client.py               # RTSP 视频流 + UDP 遥测并发网络客户端
├── ui_qt/
│   └── gui_host.py                  # PyQt5 纯表现层（View 组件）
├── database/
│   └── db_manager.py                # SQLite3 告警持久化与 LRU 容量管理
├── tools/
│   ├── build_and_train_yolo.py      # IQ → VIRIDIS 瀑布图数据集 + YOLOv8 训练
│   └── convert_yolo_to_rknn.py      # YOLOv8 → RKNN FP16 离线转换工具
├── mock_transmitter/
│   ├── uav_tx_gui.py                # PlutoSDR 无人机射频靶机控制台（GUI）
│   │                                  支持机型：DJI Mini 4 Pro / Mavic 3 / Avata 2 /
│   │                                            FPV Combo（各四种带宽变体）
│   │                                  模式：跳频模式 / 单频模式（可选扇区）
│   └── mock_k230.py                 # PC 侧 K230 模拟器（MJPEG 流 + UDP 遥测）
├── deploy_orangepi.sh               # 香橙派 5 首次部署一键环境装配脚本
└── .gitignore                       # 仅排除 s3_thresholds.json 及 *.pyc
```

## 7. 部署与装配指南

```bash
# 克隆仓库
git clone https://github.com/ALPssdz/RF-Vision-UAV-Tracker.git
cd RF-Vision-UAV-Tracker

# 启动系统
# → 自动执行环境底噪校准（约 60 秒），完成后进入检测主循环
python3 system_hub.py
```

> **注意**：`rf_zynq/s3_thresholds.json` 由各设备本地生成，已排除 Git 追踪。每当部署地点改变、周围电磁环境显著变化或调整 RX 增益后，需重启 `system_hub.py` 重新校准。

### SDR 参数配置（`backend_rk3588/config.py`）

```python
SDR_URI             = "ip:192.168.31.10"        # AD9364 网络地址（按实际局域网修改）
SDR_GAIN_DB         = 70                         # MGC 接收增益（dB），AD9364 最大约 73 dB
SAMPLE_RATE         = int(40e6)                  # ADC 采样率：40 MSps
SWEEP_SECTORS       = [5745e6, 5785e6, 5825e6]  # OcuSync 5.8 GHz 三扇区中心频率（Hz）
YOLO_ASSIST_ENABLED = True                       # 已在 RFUAV 5.8 GHz 瀑布图数据集上完成训练
YOLO_CONF_THRESH    = 0.30                       # 因域偏移（RFUAV → 实际 SDR）适当降低
```

### 构建 YOLO 数据集与训练（可选 — 已附带预训练权重）

```bash
# 从 RFUAV IQ 录制生成 VIRIDIS 瀑布图数据集并训练 YOLOv8n
python tools/build_and_train_yolo.py

# 若数据集已生成，跳过生成阶段直接训练
python tools/build_and_train_yolo.py --skip-gen
```

### 将 YOLOv8 权重转换为 RKNN FP16

需在 **x86 Linux / WSL2** 环境执行（需安装 `rknn-toolkit2`），转换完成后将 `best.rknn` 复制到香橙派：

```bash
python tools/convert_yolo_to_rknn.py
scp rf_zynq/yolo/best.rknn orangepi@<IP>:/opt/RF-Vision-UAV-Tracker/rf_zynq/yolo/
```

> **注意**：INT8 量化会破坏 YOLOv8 类别置信度。转换工具使用 **FP16 模式**（`do_quantization=False`），完整保留推理精度，NPU 推理速度仅比 INT8 慢约 30%。

## 8. 实测检测性能

**2026-04-06** 实机验证，使用 PlutoSDR 发射机（DJI Mini 4 Pro / Mavic 3 IQ 数据集）+ AD9364 接收机（70 dB MGC 增益）：

| 事件 | 扇区 | 联合 NCC | 阈值 | PSR | CFS | 结果 |
|------|------|---------|------|-----|-----|------|
| OcuSync 检测 | 5785 MHz | 3.92% | 1.80% | 7.6× | 4.9× | ✅ 已确认 |
| OcuSync 检测（弱信号） | 5785 MHz | 2.02% | 1.80% | 6.0× | 5.8× | ✅ 已确认 |
| OcuSync 检测 | 5785 MHz | 3.46% | 1.80% | 10.1× | 3.5× | ✅ 已确认 |
| OcuSync 检测（强信号） | 5825 MHz | 7.85% | 3.25% | 18.4× | 2.6× | ✅ 已确认 |
| SMPS 突发（α=14.4 kHz） | 5825 MHz | 9.40% | — | 2.6× | **1.06×** | ❌ 正确剔除（CFS 失败）|
| 宽带噪声 | 5745 MHz | 1.82% | — | **2.35×** | 1.78× | ❌ 正确剔除（PSR 失败）|

> **v4.0 升级说明**：以上数据基于 v3.x 参数采集。在 v4.0 参数下（CHUNK=160k、PEAK_WEIGHT=0.65、AFS 验证、SDS 软下限），弱信号行（NCC=2.02%）将额外受益于 0.80×th 软下限救援路径；SMPS/噪声拒绝行在 CFS 阈值上调（2.0×）与 AFS（σ_α 检查）双重加固保护下，误报风险进一步降低。

**实测循环频率一致性：alpha = 27.99 kHz**，对应 OcuSync OFDM CP=1/4 物理结构（N_total = 1429 采样点 @ 40 MSps），前后多次检测结果高度一致，表明协议指纹识别稳定可靠。
