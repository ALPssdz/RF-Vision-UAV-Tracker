# RF-Vision-UAV-Tracker

**English** | [简体中文](README_zh.md)

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. System Architecture](#2-system-architecture)
- [3. Three-Stage RF Detection Pipeline (v4.0)](#3-three-stage-rf-detection-pipeline-v40)
- [4. S3 Autonomous Field Calibration](#4-s3-autonomous-field-calibration)
- [5. Asymmetric Fusion Methodology](#5-asymmetric-fusion-methodology)
- [6. Software Stack & Module Organization](#6-software-stack--module-organization)
- [7. Deployment Instructions](#7-deployment-instructions)
- [8. Validated Detection Performance](#8-validated-detection-performance)

## 1. Introduction
RF-Vision-UAV-Tracker is a distributed, multi-modal Unmanned Aerial Vehicle (UAV) detection and early-warning system. By integrating Software-Defined Radio (SDR) with edge-computing optical vision, this system addresses the inherent limitations of single-sensor detection methodologies (e.g., localized blind spots and vulnerability to radio silence). It employs an asymmetric Out-Of-Band (OOB) sensor fusion architecture to achieve robust target acquisition and evidentiary logging in complex electromagnetic environments.

The central controller runs on **Orange Pi 5 (RK3588)**, leveraging the onboard **NPU (Neural Processing Unit)** via RKNN-Toolkit2 to execute hardware-accelerated YOLOv8 inference on the RF spectrogram stream, delivering significantly better real-time performance than CPU-only inference.

## 2. System Architecture
The hardware topology is established upon a Gigabit Ethernet LAN, interconnecting three decoupled physical nodes:

*   **RF Sensing Node (ZYNQ-7020 + AD9364)**
    Primary omnidirectional detection array. Leverages the 56 MHz tuning bandwidth of the AD9364 transceiver with vertically polarized dual-band antennas to sweep the 5.8 GHz ISM band (DJI OcuSync channels). Streams IQ samples over TCP/IP to the central controller via `libiio` / `pyadi-iio`.

*   **Vision Sensing Node (Kendryte K230)**
    Zenith-compensation node equipped with a 1080P optical sensor and onboard KPU for hardware-accelerated YOLO inference. Offsets the RF antenna "Zenith Null" (overhead polarization blind spot). Sends video via RTSP and lightweight alert telemetry (bounding box + confidence) via a stateless UDP side-channel.

*   **Central Controller (Orange Pi 5 — RK3588)**
    Global event bus and aggregation hub. Executes the three-stage RF detection pipeline, runs YOLOv8 spectrogram inference on the RK3588 NPU via RKNN-Toolkit-Lite2, fuses multi-modal evidence, and serves a PyQt5 GUI with real-time visualization and SQLite3 alert persistence.

## 3. Three-Stage RF Detection Pipeline (v4.0)

```
IQ Samples (AD9364, 40 MSps, 2.62M samples/burst)
        │
        ▼
  Stage 1 — Kurtosis-Weighted RSSI Pre-scan (S1)  [v4.0]
    Fast power measurement across all 5.8 GHz sectors.
    Kurtosis-weighted priority ranking:
      P̃_f = P̄_f · (1 + β·(κ_f − 3)/3),  β = 0.40
    OcuSync burst frames (κ ≈ 6~8 vs. noise κ ≈ 3) are amplified
    by 40~80% in the sector score, ensuring weak-burst sectors are
    selected for deep analysis even at low average power.
    Buffer: 524,288 samples (13.1 ms); 3-frame median filtering.
        │
        ▼
  Stage 2 — Spectrogram + YOLOv8 (S2)
    STFT waterfall image (640×640, HOT colormap).
    YOLOv8n inference on RK3588 NPU via RKNN (~30 ms).
    bbox_score forwarded to alert_info for optional SDS injection.
    [Note: YOLO assist injection is OFF by default until the model
     is retrained on 5.8 GHz waterfall data. See config.py.]
        │
        ▼
  Stage 3 — Cyclic Frequency Discriminator v4.0 (S3)
    Four-Layer Triple Orthogonal Verification + Soft Decision Scoring.

    CAF-FFT core (inherited):
      z[n] = x[n]·x*[n-τ],  NCC[α] = |FFT(z)[k]| / (N_z · P_x)
      Wi-Fi leakage into OcuSync channel (N=160 000):
        NCC_WiFi ≈ P_WiFi · sinc(1130) ≈ 0.028% of Wi-Fi power

    Target cycle frequencies (Fs = 40 MSps):
      OcuSync 2.0  (Δf=15 kHz, τ=2667):     α_sym ≈ 10.5–14.5 kHz
      OcuSync 3.0/4.0 (Δf=30 kHz, τ=1333): α_sym ≈ 22–30 kHz
      Wi-Fi 802.11 (Δf=312.5 kHz, τ=128):  α_sym = 250 kHz  ← orthogonal

    v4.0 Decision Engine — Soft Decision Scoring (SDS):
      S = 0.45·(NCC/th) + 0.25·log₁₀(PSR/th_PSR)
        + 0.20·log₁₀(CFS/th_CFS) + 0.10·I[AFS_pass]
      DETECT if S ≥ 1.0  AND  NCC ≥ 0.80×th  (soft floor)
      BYPASS if NCC ≥ 2.5×th  (strong signal, no SDS required)

    Four orthogonal verification layers:
      L1 — Frame CAF-FFT  (CHUNK=160k, OVERLAP=80%, PEAK_WEIGHT=0.65)
      L2 — Combined statistic > per-sector adaptive threshold
      L3 — τ-domain PSR (Peak-to-Sidelobe Ratio) ≥ 2.2×
      L4 — α-domain CFS (Cyclic Frequency Sharpness) ≥ 2.0×
      AFS — Alpha Frequency Stability: σ_α < 500 Hz across frames
            (OcuSync TCXO drift <500 Hz vs. SMPS/Wi-Fi jitter ~2–5 kHz)
```

### False-Alarm Rate Model (v4.0)

| Branch | TPF N | P_fa (final) |
|--------|-------|-------------|
| Strong bypass (NCC ≥ 3×th) | 1 | < 0.10% (AFS+PSR+CFS) |
| Medium  (1.8×th ≤ NCC < 3×th) | 2 | ≈ 0.25% |
| Weak    (th ≤ NCC < 1.8×th) | 3 | ≈ 0.013% |

The Tri-Level Elastic TPF also uses **streak decay** instead of hard reset:
`streak[t+1] = max(0, streak[t] − 0.5)` — signals that briefly drop below
threshold recover their confirmation count within 2 ticks, reducing re-acquisition delay.

## 4. S3 Autonomous Field Calibration

Every time `system_hub.py` starts, it automatically executes a **zero-interaction ambient calibration** before entering detection mode. No user password or confirmation is required.

### Calibration Algorithm

For each of the three 5.8 GHz sectors, `calibrate_s3.py` captures `N = 8` IQ bursts and computes the CAF-NCC background floor:

```
bg_eff = 0.4 × bg_max + 0.6 × bg_avg          (outlier-robust weighted estimate)
th     = max(HARD_FLOOR, bg_eff × NOISE_MARGIN) (per-sector, independent)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `NOISE_MARGIN` | 2.0× | SDS funnel (PSR+CFS+AFS) provides primary false-alarm guard |
| `BG_MAX_WEIGHT` | 0.40 | Reduces single-burst SMPS spike influence |
| `HARD_FLOOR_30K` | 1.8% | 8× theoretical NCC floor (1/√160 000 ≈ 0.25%) |
| `HARD_FLOOR_15K` | 1.4% | 6× theoretical NCC floor |
| `ALPHA_SCAN_30K` | 22–30 kHz | Covers OcuSync CP variants (CP=1/8 → 26.7 kHz, CP=1/4 → 24.0 kHz) |
| `ALPHA_SCAN_15K` | 10.5–14.5 kHz | Covers OcuSync 15 kHz channel variants |

Calibrated thresholds are written to `rf_zynq/s3_thresholds.json` (git-ignored) and loaded dynamically at startup. **The source code is never patched** — no Git conflicts occur across deployments.

### Calibration Console Output (example)

```
  [Sector 5785MHz]  OcuSync 30kHz: avg=0.31%  max=0.91%
  [Sector 5745MHz]  OcuSync 30kHz: avg=0.55%  max=1.21%
  [Sector 5825MHz]  OcuSync 30kHz: avg=1.04%  max=2.50%

  Per-sector derived thresholds  (NOISE_MARGIN = 2.0×)
  5785 MHz:  TH_30k= 1.80%   TH_15k= 1.40%   ← hard floor (clean sector)
  5745 MHz:  TH_30k= 2.45%   TH_15k= 1.40%
  5825 MHz:  TH_30k= 3.25%   TH_15k= 1.95%
```

## 5. Asymmetric Fusion Methodology
Differing from conventional Boolean AND-logic fusion, this system implements independent, asynchronous trigger paths to maximize detection recall:

1.  **RF Trigger (Primary)** — S3 SDS discriminator confirms OcuSync protocol fingerprint through four orthogonal verifications and fires an alert with a cyclic spectrum snapshot.
2.  **Visual Trigger (Secondary)** — K230 UDP telemetry independently triggers an alert, compensating for UAVs under RF silence or traversing the antenna null.
3.  **YOLO Assist (Optional, disabled by default)** — Once retrained on 5.8 GHz waterfall data (`YOLO_ASSIST_ENABLED = True` in config.py), S2 bbox_score injects +0.15 into the SDS score for weak-signal rescue when S3 SDS ∈ [0.85, 1.00).

Both primary trigger paths produce a fused composite evidence image (RF waterfall + optical frame) stored in the SQLite3 alert database.

## 6. Software Stack & Module Organization

```
RF-Vision-UAV-Tracker/
├── system_hub.py                    # Entry point & central pipeline orchestrator
├── backend_rk3588/
│   ├── config.py                    # Centralized hardware config + YOLO_ASSIST_ENABLED flag
│   └── main_rf_pipeline.py          # RFToolchain: S1→S2→S3 pipeline controller (v4.0)
├── rf_zynq/
│   ├── rf_stage1_rssi_scan.py       # S1: Kurtosis-weighted fast RSSI scan (v4.0)
│   ├── rf_stage2_waterfall_yolo.py  # S2: IQ → STFT waterfall tensor
│   ├── rf_stage3_cyclostationary.py # S3: CAF-FFT + AFS + SDS discriminator (v4.0)
│   ├── calibrate_s3.py              # S3 autonomous field calibration wizard
│   ├── s3_thresholds.json           # Runtime thresholds (git-ignored, auto-generated)
│   └── rknn_infer.py                # RKNN-Lite2 YOLOv8 NPU inference wrapper
├── vision_k230/
│   └── k230_client.py               # RTSP video + UDP telemetry network client
├── ui_qt/
│   └── gui_host.py                  # PyQt5 presentation layer (View only)
├── database/
│   └── db_manager.py                # SQLite3 alert persistence & LRU management
├── tools/
│   └── convert_yolo_to_rknn.py      # YOLOv8 → RKNN INT8 offline converter
├── mock_transmitter/
│   ├── uav_tx_gui.py                # PlutoSDR UAV RF target simulator GUI
│   │                                  Supports: DJI Mini 4 Pro / Mavic 3 / Avata 2 /
│   │                                            FPV Combo (4 bandwidth variants each)
│   │                                  Modes: frequency-hopping OR single-sector
│   └── mock_k230.py                 # PC-side K230 simulator (MJPEG + UDP)
├── deploy_orangepi.sh               # One-shot Orange Pi 5 environment setup
└── .gitignore                       # Excludes s3_thresholds.json & *.pyc only
```

## 7. Deployment Instructions

```bash
# Clone the repository
git clone https://github.com/ALPssdz/RF-Vision-UAV-Tracker.git
cd RF-Vision-UAV-Tracker

# Launch the system
# → Automatic ambient calibration runs first (~60 s), then detection begins
python3 system_hub.py
```

> **Note**: `rf_zynq/s3_thresholds.json` is created locally on each device and is excluded from Git. Re-run `system_hub.py` whenever the deployment environment changes (new location, changed interference floor, RX gain adjustment).

### Adjust SDR Parameters (`backend_rk3588/config.py`)

```python
SDR_URI           = "ip:192.168.31.10"         # AD9364 network address
SDR_GAIN_DB       = 70                          # MGC RX gain (dB); AD9364 max ≈ 73 dB
SAMPLE_RATE       = int(40e6)                   # 40 MSps
SWEEP_SECTORS     = [5745e6, 5785e6, 5825e6]   # OcuSync 5.8 GHz band sectors
YOLO_ASSIST_ENABLED = False                     # Enable after retraining on 5.8 GHz data
```

### (Optional) Convert YOLOv8 Weights to RKNN INT8

Run on **x86 Linux / WSL2**, then copy `best.rknn` to the Orange Pi:

```bash
python tools/convert_yolo_to_rknn.py
```

## 8. Validated Detection Performance

Field-validated on **2026-04-06** with PlutoSDR mock transmitter (DJI Mini 4 Pro / Mavic 3 IQ dataset) and AD9364 receiver at 70 dB MGC gain:

| Event | Sector | Combined NCC | Threshold | PSR | CFS | Result |
|-------|--------|-------------|-----------|-------|-------|--------|
| OcuSync detected | 5785 MHz | 3.92% | 1.80% | 7.6× | 4.9× | ✅ CONFIRMED |
| OcuSync detected (weak) | 5785 MHz | 2.02% | 1.80% | 6.0× | 5.8× | ✅ CONFIRMED |
| OcuSync detected | 5785 MHz | 3.46% | 1.80% | 10.1× | 3.5× | ✅ CONFIRMED |
| OcuSync detected | 5825 MHz | 7.85% | 3.25% | 18.4× | 2.6× | ✅ CONFIRMED |
| SMPS burst (α=14.4 kHz) | 5825 MHz | 9.40% | — | 2.6× | **1.06×** | ❌ Rejected (CFS) |
| Wideband noise | 5745 MHz | 1.82% | — | **2.35×** | 1.78× | ❌ Rejected (PSR) |

> **v4.0 upgrade note**: The above results were obtained with v3.x parameters. With v4.0 (CHUNK=160k, PEAK_WEIGHT=0.65, AFS guard, SDS soft floor), the weak-signal row (NCC=2.02%) now benefits from the 0.80×th soft floor path, and the SMPS/noise rejection rows are additionally guarded by AFS (σ_α check) and the stricter CFS threshold (2.0×).

**Consistent detected alpha: 27.99 kHz** → corresponds to OcuSync OFDM symbol structure with CP=1/4 at 35 kHz subcarrier spacing (N_total = 1429 samples @ 40 MSps).