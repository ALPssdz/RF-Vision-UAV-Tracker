"""
主射频检测流水线（RF Detection Pipeline）
==========================================
实现三级级联检测架构（S1 → S2 → S3）：

  Stage 1 (S1): 快速 RSSI 预扫（Fast Power Pre-scan）
    — 对所有扫描扇区进行宽带功率测量，以最小时间开销确定信号占优扇区

  Stage 2 (S2): 短时驻留与频谱成像（Dwell & Spectrogram Generation）
    — 在 S1 选定扇区采集高时间分辨率 IQ 数据，生成 640×640 短时傅里叶变换瀑布图
    — 同时运行 YOLOv8 视觉推理（当前为辅助显示，待 5.8GHz 数据集重训后恢复门控功能）

  Stage 3 (S3): 循环谱物理层审计（Cyclostationary Physical-Layer Audit）
    — 基于 OcuSync 协议的 OFDM 循环前缀时延特征进行协议指纹识别
    — 独立运行，不依赖 S2 YOLO 推理结果（5.8GHz 场景下 YOLO 视觉门控暂时旁路）
"""

from rf_zynq.rf_stage1_rssi_scan import RF_Stage1_RSSIScan
from rf_zynq.rf_stage2_waterfall_yolo import RF_Stage2_Dwell
from rf_zynq.rf_stage3_cyclostationary import RF_Stage3_CycloAudit
import time


def load_yolo_model():
    """
    动态搜索并加载最新训练权重文件（best.pt）。

    搜索策略：
      通配符匹配 runs/*/weights/best.pt 和 runs/detect/*/weights/best.pt，
      按文件修改时间排序，优先加载最新一次训练的权重。
    """
    from ultralytics import YOLO
    import os, glob

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_root    = os.path.join(project_root, "rf_zynq", "yolo", "runs")

    patterns = [
        os.path.join(runs_root, "*", "weights", "best.pt"),
        os.path.join(runs_root, "detect", "*", "weights", "best.pt"),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))

    if not matches:
        raise FileNotFoundError(f"未找到 YOLO 权重文件，请检查训练输出路径: {runs_root}")

    best_model_path = sorted(matches, key=os.path.getmtime)[-1]
    print(f"[YOLO] 已加载权重: {best_model_path}")
    return YOLO(best_model_path)


def active_yolo_inference(model, tensor_bgr):
    """
    对输入频谱张量执行 YOLOv8 目标检测推理。

    Parameters
    ----------
    tensor_bgr : ndarray
        640×640×3 BGR 格式频谱瀑布图张量

    Returns
    -------
    (bool, float, ndarray) : (检测标志, 最高置信度, 标注图像)
      注：当前该函数的检测结果仅用于 UI 显示，不参与 S3 触发逻辑。
          原因：现有权重基于 2.4 GHz 频谱数据训练，在 5.8 GHz 场景下识别率不可靠。
    """
    import numpy as np
    results = model.predict(source=tensor_bgr, verbose=False)

    highest_score = 0.0
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            highest_score = float(np.max(confs))

    is_detected   = highest_score > 0.60
    annotated_frame = results[0].plot()
    return is_detected, highest_score, annotated_frame


class RFToolchain:
    """
    三级射频检测流水线主控类（S1-RSSI → S2-YOLO → S3-CycloAudit）

    tick() 方法执行一次完整的检测周期：
      1. S1 快速 RSSI 预扫，获取各扇区信号功率排名
      2. 将 SDR 调谐至功率最强扇区
      3. S2 采集高分辨率 IQ 数据并生成频谱瀑布图，YOLOv8 辅助推理
      4. S3 对同一段 IQ 数据执行循环谱审计，输出最终告警判决

    硬件配置：
      - SDR 前端：ZYNQ7020 + AD9364，采样率 40 MSps，人工增益 50 dB
      - 扫描频段：5725~5845 MHz（覆盖 DJI OcuSync 5.8 GHz 全频段）
      - 缓冲区：2,621,440 采样点 = 65.5 ms 连续无缝时间切片
    """

    def __init__(self, uri="ip:192.168.31.10"):
        import adi
        try:
            self.sdr = adi.Pluto(uri)
            self.sample_rate = int(40e6)
            self.sdr.sample_rate      = self.sample_rate
            self.sdr.rx_rf_bandwidth  = self.sample_rate

            # 接收缓冲区大小：同时满足 640×4096 STFT 矩阵需求与 S3 循环谱统计精度要求
            self.sdr.rx_buffer_size = 2621440

            # 手动增益控制（MGC）：禁用 AGC，固定接收增益为 50 dB
            # 目的：保持与训练数据集一致的动态范围，防止 AGC 在低信号时将背景噪声放大
            self.sdr.rx_hardwaregain_control_mode = 'manual'
            self.sdr.rx_hardwaregain_chan0        = 50
            print("[INFO] RFToolchain: AD9364 SDR 前端初始化完成（手动增益 50 dB）。")
        except Exception as e:
            print(f"[ERROR] RFToolchain: SDR 初始化失败: {e}")
            raise e

        self.brain_yolo    = load_yolo_model()
        self.stage2_vision = RF_Stage2_Dwell(self.sdr)
        self.stage3_audit  = RF_Stage3_CycloAudit(sample_rate=self.sample_rate)

        # 扫描扇区中心频率（覆盖 DJI Mini 3 OcuSync 5.8 GHz 全频段）
        # 各扇区带宽 = 采样率 = 40 MHz，三扇区合计覆盖 5725~5845 MHz
        self.sweep_sectors = [5745e6, 5785e6, 5825e6]

        # S1 快速 RSSI 预扫模块
        self.stage1_rssi = RF_Stage1_RSSIScan(self.sdr, self.sweep_sectors, self.sample_rate)

        self.cycle_count = 0

    def tick(self):
        """
        执行一次完整的三级检测周期。

        Returns
        -------
        (ndarray, str, bool, dict) :
          - annotated_frame : 带标注的频谱瀑布图（640×640×3 BGR）
          - log_text        : 本周期诊断日志文本
          - alert_flag      : OcuSync 告警标志
          - alert_info      : 告警附属信息（频点、得分）
        """
        self.cycle_count += 1
        log_lines = []

        # ── Stage 1：RSSI 快速功率预扫 ────────────────────────────────────────
        # 对全部扇区采集短时 IQ 数据（各 6.5 ms），测量功率谱密度均值，
        # 选择接收功率最大的扇区作为 S2/S3 的分析目标，
        # 将算力集中在信号占优扇区，缩短有效检测延迟约 2~3 倍。
        ranked_sectors    = self.stage1_rssi.scan_and_rank()
        active_center_freq, rssi_top = ranked_sectors[0]

        log_lines.append(
            f"\n===== [Cycle {self.cycle_count}] "
            f"S1 优先扇区: {active_center_freq/1e6:.0f} MHz "
            f"(P_rx = {rssi_top*1e6:.2f} μW) ====="
        )

        # ── Stage 2：LO 调谐与频谱成像 ────────────────────────────────────────
        # 将本振（LO）调谐至目标扇区中心频率，等待 PLL 锁定后销毁残余缓冲，
        # 采集 65.5 ms 连续 IQ 数据，执行 FFT（N=4096）生成时频瀑布图。
        time_tune = time.time()
        self.sdr.rx_lo = int(active_center_freq)
        time.sleep(0.04)              # PLL 锁定等待（AD9364 规格 < 10 ms，保守取 40 ms）
        self.sdr.rx_destroy_buffer()  # 清除 LO 切换过渡期间的残余 IQ 数据
        log_lines.append(
            f"[S2] LO 调谐至 {active_center_freq/1e6:.0f} MHz，"
            f"缓冲区已刷新，耗时 {time.time()-time_tune:.3f} s"
        )

        time_s2 = time.time()
        waterfall_tensor = self.stage2_vision.generate_waterfall_tensor(active_center_freq)
        yolo_flag, bbox_score, annotated_frame = active_yolo_inference(self.brain_yolo, waterfall_tensor)
        cost_s2 = time.time() - time_s2

        import cv2
        cv2.putText(annotated_frame, f"SECTOR: {active_center_freq/1e6:.0f} MHz",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        log_lines.append(
            f"[S2] 频谱瀑布图生成完毕，耗时 {cost_s2:.3f} s | "
            f"YOLO 置信度: {bbox_score:.4f}（当前为辅助显示模式）"
        )

        alert_flag = False
        alert_info = {}

        # ── Stage 3：循环谱物理层审计 ─────────────────────────────────────────
        # 注：S3 当前以独立模式运行，不依赖 YOLO 推理结果触发，
        #     原因：现有 YOLO 权重基于 2.4 GHz 训练数据，在 5.8 GHz 频段视觉识别不可靠。
        #     待采集 5.8 GHz 频谱数据并重新训练后，将恢复为 if yolo_flag 串联判决架构。
        time_s3 = time.time()
        confirm_flag, audit_score = self.stage3_audit.run_spectral_audit(
            self.stage2_vision.last_buffer_iq
        )
        cost_s3 = time.time() - time_s3
        log_lines.append(
            f"[S3] 循环谱审计完成，耗时 {cost_s3:.3f} s | "
            f"YOLO={yolo_flag}({bbox_score:.3f}) | "
            f"S3={'DETECTED' if confirm_flag else 'NEGATIVE'} (score={audit_score:.4f})"
        )

        if confirm_flag:
            log_lines.append(
                f"[ALERT] OcuSync 协议特征确认！"
                f"频点: {active_center_freq/1e6:.0f} MHz | "
                f"归一化自相关系数: {audit_score:.4f}"
            )
            alert_flag = True
            alert_info = {"freq_mhz": active_center_freq / 1e6, "score": audit_score}
            cv2.putText(annotated_frame, f"S3 LOCK  score={audit_score:.3f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            log_lines.append(f"[S3] 未检测到 OcuSync 特征，当前扇区判定为无威胁目标。")

        if alert_flag:
            log_lines.append(
                "<span style='color: #ff3333; font-weight: bold;'>"
                "【最终判决】: 高置信度告警 — 检测到疑似无人机射频信号！"
                "</span>"
            )
        else:
            log_lines.append("【最终判决】: 当前扇区无异常射频活动。")

        return annotated_frame, "\n".join(log_lines), alert_flag, alert_info
