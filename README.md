# RF-Vision-UAV-Tracker

**English** | [简体中文](README_zh.md)

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. System Architecture](#2-system-architecture)
- [3. Asymmetric Fusion Methodology](#3-asymmetric-fusion-methodology)
- [4. Software Stack & Module Organization](#4-software-stack--module-organization)
- [5. Deployment Instructions](#5-deployment-instructions)

## 1. Introduction
The RF-Vision-UAV-Tracker is a distributed, multi-modal Unmanned Aerial Vehicle (UAV) detection and early-warning system. By integrating Software-Defined Radio (SDR) with edge-computing optical vision, this system addresses the inherent limitations of single-sensor detection methodologies (e.g., localized blind spots and vulnerability to radio silence). It employs an asymmetric Out-Of-Band (OOB) sensor fusion architecture to achieve robust target acquisition and evidentiary logging in complex environments.

## 2. System Architecture
The hardware topology is established upon a Gigabit Ethernet Local Area Network (LAN), interconnecting three highly decoupled physical nodes:

*   **RF Sensing Node (ZYNQ-7020 + AD9364)**
    Functions as the primary omnidirectional detection array. It leverages the 56 MHz tuning bandwidth of the AD9364 transceiver alongside vertically polarized dual-band antennas to execute continuous spectrum sweeping across the 2.4 GHz and 5.8 GHz ISM bands.
*   **Vision Sensing Node (Kendryte K230)**
    Serves as the secondary zenith-compensation node. Equipped with a 1080P optical sensor, this edge-computing module utilizes the onboard KPU (Knowledge Processing Unit) for hardware-accelerated YOLO inference. It offsets the inherent "Zenith Null" (overhead polarization blind spot) of the RF antennas.
*   **Central Controller (RK3588 Platform)**
    Acts as the primary event bus and aggregation hub. It executes the Python-based central routing logic, handles network I/O payload parsing, and orchestrates the synchronization of the multi-modal data streams for real-time PyQt5-based visualization and database persistence.

## 3. Asymmetric Fusion Methodology
Differing from conventional Boolean AND-logic sensor fusion, this system implements an independent, asynchronous trigger mechanism to ensure maximum detection recall:

1.  **RF Feature Extraction (Primary Trigger)**
    The system identifies frequency-hopping or continuous-wave patterns indicative of UAV data links. A positive anomaly match triggers a localized system alert.
2.  **Visual Telemetry (Secondary Trigger)**
    To mitigate delays associated with high-resolution video streams, the K230 node implements an Out-Of-Band (OOB) communication protocol. Video data is transferred via standard RTSP/TCP, while lightweight inference outcomes (bounding boxes, confidence scores) are transmitted via a stateless UDP side-channel. A positive optical lock independently triggers an alert, compensating for UAVs operating under radio silence or traversing the RF zenith null.

## 4. Software Stack & Module Organization
Designed with strict adherence to Model-View-Controller (MVC) paradigms, the repository is logically segmented:

*   `system_hub.py`: The single entry point and central pipeline orchestrator.
*   `rf_zynq/`: Hardware abstraction layer and DSP pipeline for SDR communication.
*   `vision_k230/`: Decoupled network client managing synchronous RTSP streams and asynchronous UDP telemetry.
*   `ui_qt/`: Pure Presentation Layer (View component) restricted from accessing underlying data generation states.
*   `database/`: File abstraction layer utilizing SQLite3 for retaining timestamped composite evidence matrices.

## 5. Deployment Instructions
Ensure execution occurs from the project root directory.

### 5.1 Environment Prerequisites
```bash
pip install pyadi-iio PyQt5 opencv-python numpy torch ultralytics
```

### 5.2 System Initialization
```bash
python system_hub.py
```
Upon initialization, the underlying hardware interfaces will establish connections prior to mounting the graphical user interface.