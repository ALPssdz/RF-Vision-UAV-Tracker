RF-Vision-UAV-Tracker
**多模态无人机监测与预警系统 (基于分布式边缘计算架构)**

![C++](https://img.shields.io/badge/C++-11%2F14-blue.svg)
![Qt](https://img.shields.io/badge/Qt-5.15%2B-green.svg)
![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange.svg)
![YOLO](https://img.shields.io/badge/Vision-YOLOv8-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 📖 项目简介
本项目是一套旨在解决低空无人机（如 2.4GHz / 5.8GHz 频段消费级无人机）非法飞行与安全隐患的综合监测预警系统。
系统采用“端-边”协同的分布式计算网络，通过 **SDR 射频特征识别** 与 **边缘 AI 视觉检测 (YOLO)** 的双重融合，实现高准确率的无人机发现、识别、报警与日志取证。

## ⚙️ 核心硬件架构
系统由三个高度解耦的物理节点构成，通过 5 口千兆工业级交换机实现高速互联：
* **📻 射频感知节点 (ZYNQ7020 + AD9363)**：负责底层电磁环境扫描，利用 FPGA 进行高速 FFT 频谱分析，通过 UDP 协议低延迟外发频域特征。
* **👁️ 视觉感知节点 (Kendryte K230)**：部署轻量级 YOLO 模型，利用 KPU 硬件加速实现目标检测，同步输出 JSON 语义结果与硬件压缩的高清实时视频流。
* **🧠 中央主控与数据平台 (RK3588)**：配备 8GB RAM 与 128G NVMe SSD。运行完整版 Ubuntu，负责多路并发数据接收、时空对齐融合判定、MySQL 数据持久化及运行高性能 C++ Qt 上位机监控界面。

## 🛠️ 技术栈
* **主控与上位机 (C++ / Qt)**：原生 Qt5/Qt6、QCustomPlot (极速瀑布图渲染)、OpenCV C++ (视频流解码)。
* **射频底层 (FPGA / C)**：Vivado / Vitis、LwIP 网络协议栈、FFT IP核。
* **视觉算法 (Python / C)**：PyTorch (模型训练)、K230 KPU SDK (模型量化与部署)、流媒体推流协议。
* **数据存储 (SQL)**：MySQL (读写分离、高频并发落盘)。

## 📁 仓库目录结构
为了保证 8 人团队的代码不产生冲突，本仓库采用模块化目录管理：

```text
SkySentinel-Edge/
├── 📁 rf_zynq/             # 射频基带组代码
├── 📁 vision_k230/         # 视觉AI组代码
├── 📁 backend_rk3588/      # 主控逻辑组代码
├── 📁 ui_qt/               # 上位机监控组代码
├── 📁 database/            # 数据库组代码
├── 📁 docs/                # 项目文档
└── 📄 README.md            # 项目说明文件
