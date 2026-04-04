from rf_zynq.rf_stage1_sweeper import RF_Stage1_Sweeper
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
    核心射频处理控制管道的类封装实现，整合三阶梯级的认知检测算法流程。
    """
    def __init__(self):
        self.stage1_scan = RF_Stage1_Sweeper()
        self.stage1_scan.initialize_sdr()
        
        self.brain_yolo = load_yolo_model()
        
        self.stage2_vision = RF_Stage2_Dwell(self.stage1_scan.sdr)
        self.stage3_audit = RF_Stage3_CycloAudit(sample_rate=self.stage1_scan.sample_rate)
        
        self.cycle_count = 0

    def tick(self):
        """
        发起一次完整的宏观软硬件联合周期巡检调用。
        返回: 带有边框标注的光栅图像矩阵，状态日志组，以及告警判定布尔级标量及附带属性字典。
        """
        self.cycle_count += 1
        log_lines = []
        log_lines.append(f"\n======== [System Cycle Execution: {self.cycle_count}] ========")
        
        # [第一级触发检测梯次: 射频底噪宽带能量谱切片扫描]
        time_s1 = time.time()
        active_center_freq = self.stage1_scan.run_sweep_cycle()
        cost_s1 = time.time() - time_s1
        log_lines.append(f"[S1 - 频段能量扫描提取]: 执行耗时 {cost_s1:.2f} 秒 | 判定中心峰值候选频频率: {active_center_freq/1e6} MHz")
        
        # [第二级触发检测梯次: 锁相凝视捕获及视觉张量推断分类]
        time_s2 = time.time()
        waterfall_tensor = self.stage2_vision.generate_waterfall_tensor(active_center_freq)
        yolo_flag, bbox_score, annotated_frame = active_yolo_inference(self.brain_yolo, waterfall_tensor)
        cost_s2 = time.time() - time_s2
        log_lines.append(f"[S2 - 目标凝视锁定特征提取]: 执行耗时 {cost_s2:.2f} 秒 | 目标标量识别判定状态: {yolo_flag} (网络置信度评价: {bbox_score:.4f})")
        
        alert_flag = False
        alert_info = {}
        
        # [第三级触发检测梯次: 基于高层抽象谱学的二次假阳性审计复刻]
        if yolo_flag:
            log_lines.append("张量判定约束触发跨层协议通过，正移交 S3 模块请求基带循环谱特征校验...")
            time_s3 = time.time()
            confirm_flag, audit_score = self.stage3_audit.run_spectral_audit(self.stage1_scan.sdr)
            cost_s3 = time.time() - time_s3
            log_lines.append(f"[S3 - 底层协议级数理特征审计]: 矩阵收敛耗时 {cost_s3:.2f} 秒 | 谱学核验判定结果: {confirm_flag} (幅值量级量度: {audit_score:.4f})")
            
            if confirm_flag:
                log_lines.append(f"CRITICAL [True Positive]: 基于三轴向立体特征判定锁定入侵信源！记录频率节点 {active_center_freq/1e6} MHz。")
                alert_flag = True
                alert_info = {"freq_mhz": active_center_freq / 1e6, "score": bbox_score}
                
                import cv2
                cv2.putText(annotated_frame, "CONFIRMED: UAV SIGNAL", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)           
            else:
                log_lines.append(f"SYSTEM [False Positive Subdued]: 数理判决标定系统发现 OFDM 结构调制泛型特征环境载波噪音，截断预警投递。")
                
        if alert_flag:
            log_lines.append("【最终系统综合判定结论】: 判定：检测到无人机")
        else:
            log_lines.append("【最终系统综合判定结论】: 判定：未检测到无人机")
                
        return annotated_frame, "\n".join(log_lines), alert_flag, alert_info
