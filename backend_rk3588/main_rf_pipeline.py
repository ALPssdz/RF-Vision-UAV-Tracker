from rf_zynq.rf_stage2_waterfall_yolo import RF_Stage2_Dwell
from rf_zynq.rf_stage3_cyclostationary import RF_Stage3_CycloAudit
import time

def load_yolo_model():
    from ultralytics import YOLO
    import os, glob
    
    # 动态定位项目物理根目录路径，寻址提取最新的预训练权重体系
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_path = os.path.join(project_root, "rf_zynq", "yolo", "runs", "detect", "*", "weights", "best.pt")
    matches = glob.glob(search_path)
    
    if not matches:
        raise FileNotFoundError("YOLO pre-trained weights 'best.pt' not found in expected directory.")
        
    best_model_path = sorted(matches, key=os.path.getmtime)[-1]
    return YOLO(best_model_path)

def active_yolo_inference(model, tensor_bgr):
    """
    针对输入的 640x640 瀑布流张量数据矩阵，挂载神经网络实施置信度检测算子推断。
    """
    import numpy as np
    results = model.predict(source=tensor_bgr, verbose=False)
    
    highest_score = 0.0
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # 提取边界框最大置信度
            confs = boxes.conf.cpu().numpy()
            highest_score = float(np.max(confs))
            
    # 动态置信度判定阈值
    is_detected = highest_score > 0.60
    annotated_frame = results[0].plot()
    
    return is_detected, highest_score, annotated_frame

class RFToolchain:
    """
    极简全链路视觉轮询中枢（Round-Robin S2+S3 架构）。
    废弃 S1 的能量盲猜，由中心直接驱动双频带雷达式切片式驻留，交由 YOLO 视觉裁决。
    """
    def __init__(self, uri="ip:192.168.31.10"):
        import adi
        try:
            self.sdr = adi.Pluto(uri)
            self.sample_rate = int(40e6)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.rx_rf_bandwidth = self.sample_rate
            self.sdr.rx_buffer_size = 16384
            
            # [全局降饱和物理防爆盾] 锁死 0dB
            self.sdr.rx_hardwaregain_control_mode = 'manual'
            self.sdr.rx_hardwaregain_chan0 = 0
            print("[INFO] RFToolchain: PlutoSDR Engine Hardware Initialized Successfully. LNA protected.")
        except Exception as e:
            print(f"[ERROR] RFToolchain: SDR Initialization Failed: {e}")
            raise e
            
        self.brain_yolo = load_yolo_model()
        
        self.stage2_vision = RF_Stage2_Dwell(self.sdr)
        self.stage3_audit = RF_Stage3_CycloAudit(sample_rate=self.sample_rate)
        
        # 建立硬核的“相控阵交替扫描扇区”
        self.sweep_sectors = [2420e6, 2460e6]
        self.current_sector_idx = 0
        
        self.cycle_count = 0

    def tick(self):
        """
        发起一次雷达轮询截面推断。
        返回: 带有边框标注的光栅图像矩阵，状态日志组，以及告警判定布尔级标量及附带属性字典。
        """
        self.cycle_count += 1
        log_lines = []
        
        # 取出本次轮询的物理频段目标
        active_center_freq = self.sweep_sectors[self.current_sector_idx]
        self.current_sector_idx = (self.current_sector_idx + 1) % len(self.sweep_sectors)
        
        log_lines.append(f"\n======== [Round-Robin Pulse: {self.cycle_count} | Sector: {active_center_freq/1e6} MHz] ========")
        
        # 【物理强制时空阻断】
        # 切频并彻底摧毁上一个扇区的 Linux 底层残余 USB 缓冲！
        time_tune = time.time()
        self.sdr.rx_lo = int(active_center_freq)
        time.sleep(0.04)
        self.sdr.rx_destroy_buffer()
        log_lines.append(f"[Hardware Phase-Lock]: LO 跳频至 {active_center_freq/1e6} MHz 并完成缓冲物理销毁, 耗时 {time.time()-time_tune:.2f} 秒")
        
        # [极简新梯次 1 (原 S2): 驻留绘制与视觉判决]
        time_s2 = time.time()
        waterfall_tensor = self.stage2_vision.generate_waterfall_tensor(active_center_freq)
        yolo_flag, bbox_score, annotated_frame = active_yolo_inference(self.brain_yolo, waterfall_tensor)
        cost_s2 = time.time() - time_s2
        
        # 在屏幕和日志上打上大大的频点标识，防伪识别
        import cv2
        cv2.putText(annotated_frame, f"SECTOR: {active_center_freq/1e6} MHz", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        
        log_lines.append(f"[S2 - 视觉驻留捕捉]: 执行耗时 {cost_s2:.2f} 秒 | 目标识别状态: {yolo_flag} (YOLO置信度: {bbox_score:.4f})")
        
        alert_flag = False
        alert_info = {}
        
        # [极简新梯次 2 (原 S3): OcuSync 物理层审计]
        if yolo_flag:
            log_lines.append("张量视觉触发告警！系统自动唤醒 S3 协议级特征审计器...")
            time_s3 = time.time()
            confirm_flag, audit_score = self.stage3_audit.run_spectral_audit(self.sdr)
            cost_s3 = time.time() - time_s3
            log_lines.append(f"[S3 - 物理自相关审计]: 收敛耗时 {cost_s3:.2f} 秒 | 核验结果: {confirm_flag} (基带循环特征幅值: {audit_score:.4f})")
            
            if confirm_flag:
                log_lines.append(f"CRITICAL [True Positive]: YOLO视觉与物理基带双重锁定黑飞无人机！频点: {active_center_freq/1e6} MHz。")
                alert_flag = True
                alert_info = {"freq_mhz": active_center_freq / 1e6, "score": bbox_score}
                cv2.putText(annotated_frame, "CONFIRMED: UAV LOCK!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)           
            else:
                log_lines.append(f"SYSTEM [False Positive Subdued]: S3 驳回了此次视觉告警，判定为环境干扰伪影。")
                
        if alert_flag:
            log_lines.append("<span style='color: #ff3333; font-weight: bold;'>【本帧最终判定】: 红色告警 - 猎杀网合围！</span>")
        else:
            log_lines.append("【本帧最终判定】: 扇区未发现实质性入侵。")
                
        return annotated_frame, "\n".join(log_lines), alert_flag, alert_info
