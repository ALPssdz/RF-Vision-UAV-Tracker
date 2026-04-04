# 📡 RF-Vision-UAV-Tracker
**基于多模态融合 (射频 + 边缘视觉) 的无人机监测与预警系统**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyQt](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![YOLO](https://img.shields.io/badge/Vision-YOLOv8-yellow.svg)
![SDR](https://img.shields.io/badge/Hardware-AD9364%20|%20ZYNQ-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 📖 项目简介
本项目是一套针对低空无人机（2.4GHz / 5.8GHz）等“低慢小”非法入侵目标设计的军工级分布式监控预警系统。
项目破除了传统侦测系统对单一传感器的依赖，采用 **SDR 射频特征截获 (超视距与全向探测)** 与 **K230 边缘 AI 视觉 (抗无线电静默与天顶补盲)** 相结合的**非对称双模态（OR 逻辑）防御体系**，致力于在最复杂的电磁环境下实现 100% 的准确拦截与取证。

## ⚙️ 核心硬件架构 (5口千兆局域网拓扑)
系统被划分为高度解耦的三大物理节点，全部运行于同一物理网络内：
* **📻 射频感知节点 (ZYNQ7020 + AD9364)**：作为主探测网。利用 AD9364 硬件破解版高达 56MHz 的变态级调谐带宽，配接垂直极化双频天线，对周边数公里空域进行 360 度圆柱体无死角覆盖扫频。
* **👁️ 视觉感知节点 (Kendryte K230)**：挂载 1080P 高清摄像头，主要通过百兆网口架设朝向天空的边缘节点。利用其自带的 KPU 推理模型锁定无人机，作为射频天线“天顶空洞（Zenith Null）”盲区的终端致命火控与影像固化抓取手段。
* **💻 中央控制台 (RK3588大屏上位机)**：连接 15.6 英寸 (1080P) 显示终端，运行跨线程异步 Python/PyQt5 大核代码，负责千兆口的数据高速汇聚、图像拼合及本地 SQLite 强持久化告警存证。

## 🎯 核心创新点：非对称战术融合 (Asymmetric Fusion)
我们摈弃了传统的“声光双条件 AND 验证”逻辑，转为极为鲁棒的**互补导引体制**：
1. **全域扫频 (SDR Trigger)**：一旦射频端特征比对成功，系统判断方圆几里必有遥控图传。拉响全域告警并进行底盘落网留存。
2. **天顶绝杀 (K230 Vision Trigger)**：若无人机实施静默抵近，或从 SDR 天线极化盲区（正头顶）飞过导致射频丢失，正朝天上的 K230 摄像头可瞬间独立触发二阶段斩杀警报。

## 🛠️ 技术栈
* **控制中枢与 UI**：Python 3、PyQt5 (多线程实时 GUI 渲染框架)。
* **射频基带**：`pyadi-iio` (底层硬件通讯)、NumPy (循环谱与时频瀑布图推算)。
* **视觉算法层**：OpenCV 图像拼接、PyTorch/YOLO 推理 (RKNN 模型转换适配)。
* **本地化数据库**：SQLite3 (历史截获指纹快照永久留存)。

## 📁 核心工程目录结构
为了保证工程模块互不干涉，架构划分为四大模块区域：

```text
E:\Myprojects\RF-Vision-UAV-Tracker\
├── 📁 rf_zynq/             # 射频驱控层: 与底层 AD9364 通信与数据清洗 (S1-S3 管线)
├── 📁 backend_rk3588/      # 算力控制层: 包含主核心调度器 main_rf_pipeline
├── 📁 ui_qt/               # 桌面应用层: gui_host.py 15.6寸自适应阵列前端面板
├── 📁 database/            # 数据库服务: db_manager 构建 SQLite 表单与取证档案袋
├── 📁 vision_k230/         # 光电协处理: RTSP推流接口与 K230 边缘推断桥接代码
├── 📁 alert_images/        # 自动生成的本地硬盘物证实况双模拼接热图保存区
└── 📄 rf_alert_history.db  # 系统生成的核心数据库账本
```

## 🚀 部署与使用
在主机 (Windows/RK3588 Ubuntu) 环境下，请确保进入工程最高级目录执行系统装载：
```shell
# 安装所需的依赖 (请附加其它 YOLO 相关环境)
pip install pyadi-iio PyQt5 opencv-python numpy

# 一键启动 15.6 英寸多模态雷达告警上位机
python ui_qt/gui_host.py
```