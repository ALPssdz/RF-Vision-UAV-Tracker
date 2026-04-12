# RF-Vision-UAV-Tracker

**English** | [简体中文](README_zh.md)

![Platform](https://img.shields.io/badge/Platform-Orange%20Pi%205%20%7C%20RK3588-orange)
![SDR](https://img.shields.io/badge/SDR-ZYNQ--7020%20%2B%20AD9364-blue)
![Vision](https://img.shields.io/badge/Vision-Kendryte%20K230-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A distributed, multi-modal UAV detection and early-warning system integrating Software-Defined Radio (SDR) with edge-computing optical vision. Designed to overcome the inherent limitations of single-sensor approaches (antenna zenith null, RF-silent UAVs) through an asymmetric Out-Of-Band (OOB) sensor fusion architecture.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Three-Stage RF Detection Pipeline](#three-stage-rf-detection-pipeline)
- [S3 Autonomous Field Calibration](#s3-autonomous-field-calibration)
- [Asymmetric Sensor Fusion](#asymmetric-sensor-fusion)
- [Software Module Organization](#software-module-organization)
- [Prerequisites](#prerequisites)
- [Deployment Guide](#deployment-guide)
- [Validated Detection Performance](#validated-detection-performance)

---

## System Architecture

Three decoupled physical nodes connected over a Gigabit Ethernet LAN:

```
┌─────────────────────────────────┐    TCP/IP (IQ stream)     ┌──────────────────────────────────────┐
│   RF Sensing Node               │ ────────────────────────► │   Central Controller                 │
│   ZYNQ-7020 + AD9364            │                           │   Orange Pi 5 (RK3588)               │
│   • 56 MHz tuning bandwidth     │                           │   • Three-stage RF detection pipeline │
│   • 5.8 GHz dual-band antenna   │                           │   • YOLOv8 NPU inference (RKNN FP16) │
│   • libiio / pyadi-iio          │                           │   • Multi-modal evidence fusion      │
└─────────────────────────────────┘                           │   • PyQt5 GUI + SQLite3 logging      │
                                                              └──────────────────────────────────────┘
┌─────────────────────────────────┐   RTSP video + UDP alert           ▲
│   Vision Node                   │ ──────────────────────────────────►│
│   Kendryte K230                 │
│   • 1080P optical sensor        │
│   • Onboard KPU — YOLO infer.   │
│   • Compensates antenna zenith  │
│     null (overhead blind spot)  │
└─────────────────────────────────┘
```

### Node Descriptions

| Node | Hardware | Role |
|------|----------|------|
| **RF Sensing** | ZYNQ-7020 + AD9364 | Omnidirectional 5.8 GHz IQ stream acquisition |
| **Vision** | Kendryte K230 | Zenith-null compensation via optical YOLO detection |
| **Central Controller** | Orange Pi 5 (RK3588) | Pipeline orchestration, fusion, GUI, and alert persistence |

---

## Three-Stage RF Detection Pipeline

```
IQ Samples (AD9364, 40 MSps, 2.62M samples/burst, ~65 ms)
        │
        ▼ Stage 1 — Kurtosis-Weighted RSSI Pre-scan (S1)
        │
        ▼ Stage 2 — Spectrogram Imaging + YOLOv8 NPU Inference (S2)
        │
        ▼ Stage 3 — Cyclic Frequency Discriminator (S3)
        │
        ▼ TPF — Tri-Level Elastic Temporal Persistence Filter
        │
        ▼ ALERT EVENT → SQLite3 + GUI
```

### Stage 1 — Kurtosis-Weighted RSSI Pre-scan (S1)

Fast power measurement across all 5.8 GHz sectors with burst-frame priority amplification.

**Kurtosis-weighted sector score:**

$$\tilde{P}_f = \bar{P}_f \cdot \left(1 + \beta \cdot \frac{\kappa_f - 3}{3}\right), \quad \beta = 0.40$$

OcuSync low-duty-cycle burst frames (κ ≈ 6–8) are scored **40–80% higher** than thermal noise (κ ≈ 3), ensuring weak-burst sectors are selected for deep analysis even at low average power.

| Parameter | Value |
|-----------|-------|
| Buffer size | 524,288 samples (13.1 ms) |
| Temporal filter | 3-frame median |

### Stage 2 — Spectrogram Imaging + YOLOv8 NPU Inference (S2)

Converts IQ samples into a 2D spectrogram and runs hardware-accelerated inference on the RK3588 NPU.

| Item | Detail |
|------|--------|
| Transform | STFT with Blackman window → 640×640 VIRIDIS waterfall image |
| Model | YOLOv8n, RKNN FP16 on RK3588 NPU |
| Inference latency | ~30 ms per frame |
| Training dataset | RFUAV IQ (131 recordings, 5 drone models, resampled to 40 MSps) + synthetic AWGN negatives |
| Training mAP@0.5 | **0.995** |
| Live SDR confidence | ≈ 0.2–0.7 (domain shift expected) |

The `bbox_score` output is forwarded to Stage 3 for Soft Decision Scoring (SDS) injection.

### Stage 3 — Cyclic Frequency Discriminator (S3)

Four-layer triple orthogonal verification combined with soft decision scoring.

#### CAF-FFT Core

```
z[n] = x[n] · x*[n−τ]
NCC[α] = |FFT(z)[k]| / (N_z · P_x)
```

**Wi-Fi cross-band leakage into OcuSync channel** (N = 160,000, Fs = 40 MHz):

```
NCC_WiFi ≈ P_WiFi × sinc(1130) ≈ 0.028% of P_WiFi   ← orthogonal rejection
```

**Target cyclic frequencies** (Fs = 40 MSps):

| Protocol | Subcarrier Spacing Δf | Lag τ | Cyclic Frequency α_sym |
|----------|-----------------------|-------|------------------------|
| OcuSync 2.0 | 15 kHz | 2667 | 10.5–14.5 kHz |
| OcuSync 3.0/4.0 | 30 kHz | 1333 | 22–30 kHz |
| Wi-Fi 802.11 | 312.5 kHz | 128 | 250 kHz ← orthogonal |

#### Soft Decision Scoring (SDS)

$$S = 0.45 \cdot \frac{\text{NCC}}{th} + 0.25 \cdot \log_{10}\frac{\text{PSR}}{th_\text{PSR}} + 0.20 \cdot \log_{10}\frac{\text{CFS}}{th_\text{CFS}} + 0.10 \cdot \mathbf{1}[\text{AFS pass}]$$

| Condition | Decision |
|-----------|----------|
| S ≥ 1.0 **AND** NCC ≥ 0.80 × th | ✅ DETECT (soft floor path) |
| NCC ≥ 2.5 × th | ✅ DETECT (strong-signal bypass) |
| otherwise | ❌ Reject |

#### Four Orthogonal Verification Layers

| Layer | Description | Threshold |
|-------|-------------|-----------|
| **L1** | Frame-level CAF-FFT scan | CHUNK=160k, overlap=80%, peak weight=0.65 |
| **L2** | Combined statistic vs. per-sector adaptive threshold | — |
| **L3** | τ-domain Peak-to-Sidelobe Ratio (PSR) | ≥ 2.2× |
| **L4** | α-domain Cyclic Frequency Sharpness (CFS) | ≥ 2.0× |
| **AFS** | α-domain frame-to-frame frequency stability | σ_α < 500 Hz |

> **AFS rationale:** OcuSync TCXO drift < 500 Hz; SMPS/Wi-Fi spurious peaks drift 2–5 kHz → cleanly separated.

#### Tri-Level Elastic TPF (Temporal Persistence Filter)

| Signal Strength | Confirmation Count N | Final P_fa |
|-----------------|----------------------|------------|
| Strong bypass (NCC ≥ 3×th) | 1 | < 0.10% |
| Medium (1.8×th ≤ NCC < 3×th) | 2 | ≈ 0.25% |
| Weak (th ≤ NCC < 1.8×th) | 3 | ≈ 0.013% |

Streak **decay** (not hard reset): `streak[t+1] = max(0, streak[t] − 0.5)` — brief signal dropouts recover within 2 ticks.

---

## S3 Autonomous Field Calibration

On every startup, `system_hub.py` automatically runs a **zero-interaction ambient calibration** before entering detection mode. No user input is required.

### Algorithm

For each of the three 5.8 GHz sectors, `calibrate_s3.py` acquires N = 8 IQ bursts and computes the per-sector adaptive threshold:

```
bg_eff = 0.4 × bg_max + 0.6 × bg_avg        # outlier-robust weighted estimate
th     = max(HARD_FLOOR, bg_eff × NOISE_MARGIN)  # independent per sector
```

### Calibration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `NOISE_MARGIN` | 2.0× | SDS (PSR+CFS+AFS) handles primary false-alarm rejection; threshold can be kept tight |
| `BG_MAX_WEIGHT` | 0.40 | Reduces influence of single-burst SMPS spikes |
| `HARD_FLOOR_30K` | 1.8% | 8× theoretical NCC floor (1/√160,000 ≈ 0.25%) |
| `HARD_FLOOR_15K` | 1.4% | 6× theoretical NCC floor |
| `ALPHA_SCAN_30K` | 22–30 kHz | Covers OcuSync CP=1/8 (26.7 kHz) and CP=1/4 (24.0 kHz) |
| `ALPHA_SCAN_15K` | 10.5–14.5 kHz | Covers all OcuSync 15 kHz channel variants |

Thresholds are written to `rf_zynq/s3_thresholds.json` (git-ignored, auto-generated). **Source code is never patched** — no Git conflicts across multi-device deployments.

### Example Console Output

```
[Sector 5785MHz]  OcuSync 30kHz: avg=0.31%  max=0.91%
[Sector 5745MHz]  OcuSync 30kHz: avg=0.55%  max=1.21%
[Sector 5825MHz]  OcuSync 30kHz: avg=1.04%  max=2.50%

Per-sector derived thresholds  (NOISE_MARGIN = 2.0×)
5785 MHz:  TH_30k= 1.80%   TH_15k= 1.40%   ← hard floor (cleanest sector)
5745 MHz:  TH_30k= 2.45%   TH_15k= 1.40%
5825 MHz:  TH_30k= 3.25%   TH_15k= 1.95%
```

---

## Asymmetric Sensor Fusion

Unlike conventional Boolean AND-logic fusion (requiring both sensors to fire), this system uses **independent asynchronous trigger paths** to maximize detection recall:

| Trigger | Source | Condition |
|---------|--------|-----------|
| **RF Trigger (Primary)** | S3 SDS discriminator | OcuSync protocol fingerprint confirmed via 4-layer orthogonal verification |
| **Visual Trigger (Secondary)** | K230 UDP telemetry | Independently fires; compensates RF-silent UAVs or zenith-null traversal |
| **YOLO Assist (Rescue)** | S2 bbox_score | Injects +0.15 into SDS when score ∈ [0.85, 1.00); **never triggers an alert alone** |

Both primary trigger paths produce a **composite evidence image** (RF waterfall + optical frame) stored in the SQLite3 alert database.

---

## Software Module Organization

```
RF-Vision-UAV-Tracker/
├── system_hub.py                    # Entry point & central pipeline orchestrator
│
├── backend_rk3588/
│   ├── config.py                    # Hardware config + YOLO_ASSIST_ENABLED flag
│   └── main_rf_pipeline.py          # RFToolchain: S1→S2→S3 pipeline controller
│
├── rf_zynq/
│   ├── rf_stage1_rssi_scan.py       # S1: Kurtosis-weighted fast RSSI scan
│   ├── rf_stage2_waterfall_yolo.py  # S2: IQ → STFT waterfall tensor
│   ├── rf_stage3_cyclostationary.py # S3: CAF-FFT + AFS + SDS discriminator
│   ├── calibrate_s3.py              # Autonomous field calibration wizard
│   ├── s3_thresholds.json           # Runtime thresholds (git-ignored, auto-generated)
│   └── rknn_infer.py                # RKNN-Lite2 YOLOv8 NPU inference wrapper (torch-free)
│
├── vision_k230/
│   └── k230_client.py               # RTSP video + UDP telemetry network client
│
├── ui_qt/
│   └── gui_host.py                  # PyQt5 presentation layer (View only)
│
├── database/
│   └── db_manager.py                # SQLite3 alert persistence & LRU management
│
├── tools/
│   ├── build_and_train_yolo.py      # IQ → VIRIDIS waterfall dataset + YOLOv8 training
│   └── convert_yolo_to_rknn.py      # YOLOv8 → RKNN FP16 offline converter
│
├── mock_transmitter/
│   ├── uav_tx_gui.py                # PlutoSDR UAV RF target simulator GUI
│   │                                  Models: DJI Mini 4 Pro / Mavic 3 / Avata 2 / FPV Combo
│   │                                  Modes: frequency-hopping or single-sector
│   └── mock_k230.py                 # PC-side K230 simulator (MJPEG + UDP)
│
├── deploy_orangepi.sh               # One-shot Orange Pi 5 environment setup script
└── .gitignore
```

---

## Prerequisites

### Hardware

| Component | Specification |
|-----------|---------------|
| Central Controller | Orange Pi 5 (RK3588), running Ubuntu / Debian |
| RF Node | ZYNQ-7020 FPGA board + AD9364 transceiver |
| Vision Node | Kendryte K230 development board + 1080P camera |
| Network | Gigabit Ethernet LAN (all nodes on same subnet) |

### Software (Central Controller — Orange Pi 5)

```bash
# Core dependencies
pip install pyadi-iio numpy scipy matplotlib Pillow PyQt5

# RKNN inference runtime (Lite2, no PyTorch required on-device)
# rknn-toolkit-lite2 wheel is pre-installed via deploy_orangepi.sh
```

> **Training / conversion** (x86 Linux or WSL2 only):
> ```bash
> pip install ultralytics rknn-toolkit2
> ```

---

## Deployment Guide

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ALPssdz/RF-Vision-UAV-Tracker.git
cd RF-Vision-UAV-Tracker
```

### Step 2 — Configure Hardware Parameters

Edit `backend_rk3588/config.py` to match your hardware setup:

```python
SDR_URI             = "ip:192.168.31.10"        # AD9364 IP address on LAN
SDR_GAIN_DB         = 70                         # MGC RX gain (dB); AD9364 max ≈ 73 dB
SAMPLE_RATE         = int(40e6)                  # 40 MSps
SWEEP_SECTORS       = [5745e6, 5785e6, 5825e6]  # OcuSync 5.8 GHz sector center frequencies (Hz)
YOLO_ASSIST_ENABLED = True                       # Enable S2 bbox_score → SDS injection
YOLO_CONF_THRESH    = 0.30                       # Lowered for live-SDR domain shift
```

### Step 3 — Launch the System

```bash
# Automatic ambient calibration runs first (~60 s), then detection begins
python3 system_hub.py
```

> **Note:** `rf_zynq/s3_thresholds.json` is generated locally on each device and is excluded from Git.
> Re-run `system_hub.py` whenever deployment location, RF environment, or RX gain changes.

---

### Optional — Build YOLOv8 Dataset & Train

Pre-trained weights are included. Only required if retraining with new IQ recordings.

```bash
# Generate VIRIDIS waterfall images from RFUAV IQ recordings, then train YOLOv8n
python tools/build_and_train_yolo.py

# Skip dataset generation if images are already built
python tools/build_and_train_yolo.py --skip-gen
```

### Optional — Convert YOLOv8 Weights to RKNN FP16

Run on **x86 Linux / WSL2** (requires `rknn-toolkit2`), then copy to the Orange Pi:

```bash
python tools/convert_yolo_to_rknn.py
scp rf_zynq/yolo/best.rknn orangepi@<IP>:/opt/RF-Vision-UAV-Tracker/rf_zynq/yolo/
```

> **Why FP16, not INT8?** INT8 quantization severely degrades YOLOv8 class confidence scores.
> FP16 (`do_quantization=False`) preserves full accuracy with only ~30% slower NPU inference.

---

## Validated Detection Performance

Field-validated using a PlutoSDR mock transmitter (DJI Mini 4 Pro / Mavic 3 IQ profiles) and AD9364 receiver at 70 dB MGC gain:

| Event | Sector | NCC | Threshold | PSR | CFS | Result |
|-------|--------|-----|-----------|-----|-----|--------|
| OcuSync detected | 5785 MHz | 3.92% | 1.80% | 7.6× | 4.9× | ✅ CONFIRMED |
| OcuSync detected (weak signal) | 5785 MHz | 2.02% | 1.80% | 6.0× | 5.8× | ✅ CONFIRMED |
| OcuSync detected | 5785 MHz | 3.46% | 1.80% | 10.1× | 3.5× | ✅ CONFIRMED |
| OcuSync detected (strong) | 5825 MHz | 7.85% | 3.25% | 18.4× | 2.6× | ✅ CONFIRMED |
| SMPS burst (α = 14.4 kHz) | 5825 MHz | 9.40% | — | 2.6× | **1.06×** | ❌ Rejected (CFS) |
| Wideband noise | 5745 MHz | 1.82% | — | **2.35×** | 1.78× | ❌ Rejected (PSR) |

**Consistent detected cyclic frequency: α = 27.99 kHz** → corresponds to OcuSync OFDM symbol structure with CP=1/4 at 35 kHz subcarrier spacing (N_total = 1429 samples @ 40 MSps). High repeatability across detections confirms stable protocol fingerprinting.