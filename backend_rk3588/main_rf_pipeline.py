"""
主射频检测流水线（RF Detection Pipeline）
==========================================
实现三级级联检测架构（S1 → S2 → S3）：

  Stage 1 (S1): 快速 RSSI 预扫（Fast Power Pre-scan）
    — 对所有扫描扇区进行宽带功率测量，以最小时间开销确定信号占优扇区

  Stage 2 (S2): 短时驻留与频谱成像（Dwell & Spectrogram Generation）
    — 在 S1 选定扇区采集高时间分辨率 IQ 数据，生成 640×640 短时傅里叶变换瀑布图
    — 运行 YOLOv8 目标检测（平台自适应：x86_64 用 torch，aarch64 用 RKNN NPU）

  Stage 3 (S3): 循环谱物理层审计（Cyclostationary Physical-Layer Audit）
    — 基于 OcuSync 协议的 OFDM 循环前缀时延特征进行协议指纹识别
    — 独立运行，不依赖 S2 YOLO 推理结果触发

平台支持：
  - 开发机 (x86_64)  : YOLO 使用 ultralytics + PyTorch CPU 推理
  - Orange Pi 5 (aarch64): YOLO 使用 rknn_toolkit_lite2 + RK3588 NPU 推理
"""

import os
import sys
import time
import numpy as np
import cv2

# 项目根目录加入模块搜索路径
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import config as cfg
from rf_zynq.rf_stage1_rssi_scan import RF_Stage1_RSSIScan
from rf_zynq.rf_stage2_waterfall_yolo import RF_Stage2_Dwell
from rf_zynq.rf_stage3_cyclostationary import RF_Stage3_CycloAudit


def load_yolo_model():
    """
    平台自适应 YOLO 模型加载函数。

    推理后端选择逻辑：
      config.YOLO_BACKEND == "rknn"  → RKNNLiteInfer（RK3588 NPU，约 20~40 ms/帧）
      config.YOLO_BACKEND == "torch" → ultralytics YOLO（CPU，约 150~300 ms/帧）

    RKNN 模型路径：config.YOLO_RKNN_PATH（相对于项目根目录）
    若 .rknn 文件不存在，自动降级至 torch 推理并输出警告。
    """
    if cfg.YOLO_BACKEND == "rknn":
        rknn_abs = os.path.join(PROJ_ROOT, cfg.YOLO_RKNN_PATH)
        if os.path.exists(rknn_abs):
            from rf_zynq.rknn_infer import RKNNLiteInfer
            return RKNNLiteInfer(rknn_abs, conf_thresh=cfg.YOLO_CONF_THRESH)
        else:
            print(f"[WARN] RKNN 模型未找到: {rknn_abs}")
            print(f"       请在 WSL2/Linux 上运行 tools/convert_yolo_to_rknn.py 完成转换。")
            print(f"       本次自动降级为 PyTorch CPU 推理。")

    # torch 推理路径（默认 / 降级）
    from ultralytics import YOLO
    import glob

    runs_root = os.path.join(PROJ_ROOT, "rf_zynq", "yolo", "runs")
    patterns  = [
        os.path.join(runs_root, "*", "weights", "best.pt"),
        os.path.join(runs_root, "detect", "*", "weights", "best.pt"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))

    if not matches:
        print(f"[WARN] 未找到 YOLO 权重文件（{runs_root}），S2 视觉推理将被跳过。")
        print(f"       系统将以纯 RF 模式运行（S1 + S3），告警逻辑不受影响。")
        return None

    best_pt = sorted(matches, key=os.path.getmtime)[-1]
    print(f"[YOLO] 已加载 PyTorch 权重: {best_pt}")
    return YOLO(best_pt)


def active_yolo_inference(model, tensor_bgr: np.ndarray):
    """
    对输入频谱张量执行 YOLOv8 目标检测推理。

    兼容 ultralytics YOLO 和 RKNNLiteInfer 两种后端的统一接口。

    Parameters
    ----------
    model       : YOLO 或 RKNNLiteInfer 实例
    tensor_bgr  : 640×640×3 BGR 格式频谱瀑布图张量（uint8）

    Returns
    -------
    (bool, float, ndarray) : (检测标志, 最高置信度, 标注图像)

    说明：
      当前检测结果仅用于 UI 显示，不参与 S3 告警触发逻辑。
      基于 5.8 GHz 数据重训 YOLO 后，可恢复为 S2-YOLO 串联 S3 的双重校验架构。
    """
    try:
        results = model.predict(source=tensor_bgr, verbose=False)

        # 防护：aarch64 daemon 线程中 PyTorch 首次推理可能返回 None 或空列表
        if not results:
            return False, 0.0, tensor_bgr.copy()

        highest_score = 0.0
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf.numpy()
                highest_score = float(np.max(confs))

        is_detected     = highest_score > cfg.YOLO_CONF_THRESH
        annotated_frame = results[0].plot()
        return is_detected, highest_score, annotated_frame

    except Exception as e:
        # 推理失败时返回原始张量，不阻断主循环
        print(f"[WARN] YOLO inference error (suppressed): {e}")
        return False, 0.0, tensor_bgr.copy()


class RFToolchain:
    """
    三级射频检测流水线主控类（S1-RSSI → S2-YOLO → S3-CycloAudit）

    tick() 方法执行一次完整的检测周期：
      1. S1 快速 RSSI 预扫，获取各扇区信号功率排名
      2. 将 SDR 调谐至功率最强扇区
      3. S2 采集 IQ 数据、生成频谱瀑布图，YOLO 推理（辅助显示）
      4. S3 对同一段 IQ 数据执行循环谱审计，输出最终告警判决

    硬件配置（来自 config.py）：
      SDR  : AD9364 + ZYNQ7020，采样率 40 MSps，增益 {cfg.SDR_GAIN_DB} dB
      频段 : 5725~5845 MHz（DJI OcuSync 5.8 GHz 全频段）
      缓冲 : 2,621,440 采样点 = 65.5 ms 连续时间切片
    """

    def __init__(self):
        import adi
        try:
            self.sdr = adi.Pluto(cfg.SDR_URI)
            self.sample_rate             = cfg.SAMPLE_RATE
            self.sdr.sample_rate         = self.sample_rate
            self.sdr.rx_rf_bandwidth     = self.sample_rate
            self.sdr.rx_buffer_size      = 2621440
            self.sdr.rx_hardwaregain_control_mode = 'manual'
            self.sdr.rx_hardwaregain_chan0        = cfg.SDR_GAIN_DB
            print(f"[INFO] RFToolchain: AD9364 初始化完成 "
                  f"(URI={cfg.SDR_URI}, Gain={cfg.SDR_GAIN_DB} dB, "
                  f"Backend={'RKNN-NPU' if cfg.YOLO_BACKEND == 'rknn' else 'PyTorch-CPU'})")
        except Exception as e:
            print(f"[ERROR] RFToolchain: SDR 初始化失败: {e}")
            raise

        self.brain_yolo    = load_yolo_model()
        self.stage2_vision = RF_Stage2_Dwell(self.sdr)
        self.stage3_audit  = RF_Stage3_CycloAudit(sample_rate=self.sample_rate)
        self.stage1_rssi   = RF_Stage1_RSSIScan(self.sdr, cfg.SWEEP_SECTORS, self.sample_rate)
        self.cycle_count   = 0

    def tick(self):
        """
        执行一次完整的三级检测周期。

        Returns
        -------
        (ndarray, str, bool, dict) :
          annotated_frame : 带标注的频谱瀑布图（640×640×3 BGR）
          log_text        : 本周期诊断日志
          alert_flag      : OcuSync 告警标志
          alert_info      : 告警附属信息（freq_mhz, score）
        """
        self.cycle_count += 1
        log_lines = []

        # ── Stage 1：RSSI 快速功率预扫 ────────────────────────────────────────
        ranked_sectors      = self.stage1_rssi.scan_and_rank()
        active_freq, rssi_top = ranked_sectors[0]
        log_lines.append(
            f"\n===== [Cycle {self.cycle_count}] "
            f"S1 Priority: {active_freq/1e6:.0f} MHz "
            f"(P_rx = {rssi_top*1e6:.2f} μW) ====="
        )

        # ── Stage 2：LO 调谐与频谱成像 ────────────────────────────────────────
        t0 = time.time()
        self.sdr.rx_lo = int(active_freq)
        time.sleep(0.04)               # PLL 锁定等待（40 ms）
        self.sdr.rx_destroy_buffer()
        waterfall_tensor = self.stage2_vision.generate_waterfall_tensor(active_freq)

        yolo_flag, bbox_score, annotated_frame = False, 0.0, waterfall_tensor.copy()
        if self.brain_yolo is not None:
            yolo_flag, bbox_score, annotated_frame = active_yolo_inference(
                self.brain_yolo, waterfall_tensor
            )
            backend_tag = 'RKNN-NPU' if cfg.YOLO_BACKEND == 'rknn' else 'CPU'
        else:
            backend_tag = 'DISABLED'
        cost_s2 = time.time() - t0

        cv2.putText(annotated_frame, f"SECTOR: {active_freq/1e6:.0f} MHz",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        log_lines.append(
            f"[S2] Freq={active_freq/1e6:.0f} MHz | "
            f"Cost={cost_s2:.3f} s | "
            f"YOLO={'SKIP' if self.brain_yolo is None else f'{yolo_flag}({bbox_score:.3f})'} "
            f"[{backend_tag}]"
        )

        # ── Stage 3：循环谱物理层审计 ─────────────────────────────────────────
        t1 = time.time()
        confirm_flag, audit_score = self.stage3_audit.run_spectral_audit(
            self.stage2_vision.last_buffer_iq
        )
        cost_s3 = time.time() - t1
        log_lines.append(
            f"[S3] Cost={cost_s3:.3f} s | "
            f"Result={'DETECTED' if confirm_flag else 'NEGATIVE'} "
            f"(score={audit_score:.4f})"
        )

        alert_flag = False
        alert_info = {}

        if confirm_flag:
            log_lines.append(
                f"[ALERT] OcuSync 协议特征确认！"
                f"频点: {active_freq/1e6:.0f} MHz | "
                f"归一化自相关系数: {audit_score:.4f}"
            )
            alert_flag = True
            alert_info = {"freq_mhz": active_freq / 1e6, "score": audit_score}
            cv2.putText(annotated_frame, f"S3 LOCK  score={audit_score:.3f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            log_lines.append(
                "<span style='color: #ff3333; font-weight: bold;'>"
                "【最终判决】: 高置信度告警 — 检测到疑似无人机射频信号！"
                "</span>"
            )
        else:
            log_lines.append("【最终判决】: 当前扇区无异常射频活动。")

        return annotated_frame, "\n".join(log_lines), alert_flag, alert_info
