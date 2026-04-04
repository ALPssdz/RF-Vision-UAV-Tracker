from rf_stage1_sweeper import RF_Stage1_Sweeper
from rf_stage2_waterfall_yolo import RF_Stage2_Dwell
from rf_stage3_cyclostationary import RF_Stage3_CycloAudit
import time

def load_yolo_model():
    # print("[YOLO 中枢] 尝试挂载训练好的无人机基座大模型 best.pt ...")
    from ultralytics import YOLO
    import os, glob
    
    # 动态抓取 yolo_train 文件夹下最新的 detect/train*/weights/best.pt
    search_path = os.path.join(os.path.dirname(__file__), "yolo", "runs", "detect", "*", "weights", "best.pt")
    matches = glob.glob(search_path)
    if not matches:
        raise FileNotFoundError("没有找到 best.pt，您确定模型练完了吗？")
    best_model_path = sorted(matches, key=os.path.getmtime)[-1]
    # print(f"[YOLO 中枢] 成功截获引擎点: {best_model_path}")
    return YOLO(best_model_path)

def active_yolo_inference(model, tensor_bgr):
    """
    接驳内存里从第二阶段吐过来的三通道瀑布流图彩图，执行实弹检测！
    """
    import numpy as np
    # YOLO 对于 cv2 生成的图片，默认预测
    results = model.predict(source=tensor_bgr, verbose=False)
    
    highest_score = 0.0
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # 取出最大的置信度概率分数
            confs = boxes.conf.cpu().numpy()
            highest_score = float(np.max(confs))
            
    # 高感知门限：只要有大于 60% 把握觉得像无人机，就拉警报送第三阶段
    is_detected = highest_score > 0.60
    
    # 获取 YOLO 自动帮忙画好框的彩色渲染帧！
    annotated_frame = results[0].plot()
    
    return is_detected, highest_score, annotated_frame

class RFToolchain:
    def __init__(self):
        """将之前的主函数强耦合操作，抽离为一个生命周期常驻对象"""
        self.stage1_scan = RF_Stage1_Sweeper()
        self.stage1_scan.initialize_sdr()
        
        # 加载神兵利器大模型
        self.brain_yolo = load_yolo_model()
        
        self.stage2_vision = RF_Stage2_Dwell(self.stage1_scan.sdr)
        self.stage3_audit = RF_Stage3_CycloAudit(sample_rate=self.stage1_scan.sample_rate)
        
        self.cycle_count = 0

    def tick(self):
        """
        每次调用执行一次三级审查侦测
        返回: (带有标注框的 BGR 彩色数组，日志文本，报警状态bool)
        """
        self.cycle_count += 1
        log_lines = []
        log_lines.append(f"\n======== [侦察回合: {self.cycle_count}] ========")
        
        # 【Stage 1】
        time_s1 = time.time()
        active_center_freq = self.stage1_scan.run_sweep_cycle()
        cost_s1 = time.time() - time_s1
        log_lines.append(f"📡 【S1 扫频】{cost_s1:.2f} s | 猎锁定频: {active_center_freq/1e6} MHz")
        
        # 【Stage 2】
        time_s2 = time.time()
        waterfall_tensor = self.stage2_vision.generate_waterfall_tensor(active_center_freq)
        yolo_flag, bbox_score, annotated_frame = active_yolo_inference(self.brain_yolo, waterfall_tensor)
        cost_s2 = time.time() - time_s2
        log_lines.append(f"👁️ 【S2 凝视与视觉】{cost_s2:.2f} s | AI判别: {yolo_flag} (分数 {bbox_score:.4f})")
        
        alert_flag = False
        
        # 【Stage 3】
        alert_info = {}
        if yolo_flag:
            log_lines.append("🚨 YOLO 判定威胁！进入底层 S3 审计逻辑...")
            time_s3 = time.time()
            confirm_flag, audit_score = self.stage3_audit.run_spectral_audit(self.stage1_scan.sdr)
            cost_s3 = time.time() - time_s3
            log_lines.append(f"🧠 【S3 循环谱审计】{cost_s3:.2f} s | 结论: {confirm_flag} (分数 {audit_score:.4f})")
            
            if confirm_flag:
                log_lines.append(f"🎯 [最高警报]: 发现非法入侵无人机！(频段: {active_center_freq/1e6} MHz)")
                alert_flag = True
                alert_info = {"freq_mhz": active_center_freq / 1e6, "score": bbox_score}
                
                import cv2
                cv2.putText(annotated_frame, "[S3 CONFIRM] DRONE!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)           
            else:
                log_lines.append(f"👻 [虚警排除]: 循环谱特征提取证明，这是一股强Wi-Fi信号，不予报警。")
                
        return annotated_frame, "\n".join(log_lines), alert_flag, alert_info
