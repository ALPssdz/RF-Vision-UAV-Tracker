from rf_stage1_sweeper import RF_Stage1_Sweeper
from rf_stage2_waterfall_yolo import RF_Stage2_Dwell
from rf_stage3_cyclostationary import RF_Stage3_CycloAudit
import time

def mock_yolo_inference(tensor_image):
    """
    假定接驳位于内存里的 Tensor 过一遍 RKNN / ONNX
    返回: 是否包含类似无人机的形状 (基于简单随机以作演示)
    """
    import random
    score = random.random()
    is_detected = score > 0.8 # 20%的误报率
    return is_detected, score

def main():
    print("=======================================")
    print("多模态认知无线电系统 (3-Stage) 开始运行")
    print("=======================================")
    
    # 初始化组件
    stage1_scan = RF_Stage1_Sweeper()
    stage1_scan.initialize_sdr()
    
    stage2_vision = RF_Stage2_Dwell(stage1_scan.sdr)
    stage3_audit = RF_Stage3_CycloAudit(sample_rate=stage1_scan.sample_rate)
    
    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\n--- [大循环轮次: {cycle_count}] ---")
        
        # 【Stage 1】：底层侦察与活跃段锁定
        time_s1 = time.time()
        active_center_freq = stage1_scan.run_sweep_cycle()
        cost_s1 = time.time() - time_s1
        print(f"【S1 耗时】{cost_s1:.2f} s")
        
        # 【Stage 2】：高频凝视与图像张量零拷贝组装
        time_s2 = time.time()
        waterfall_tensor = stage2_vision.generate_waterfall_tensor(active_center_freq)
        
        # 将内存数据过 YOLO
        yolo_flag, bbox_score = mock_yolo_inference(waterfall_tensor)
        cost_s2 = time.time() - time_s2
        print(f"【S2 耗时】{cost_s2:.2f} s | YOLO 判断状态: {yolo_flag}")
        
        if yolo_flag:
            print("🚨 YOLO 视觉模型拉响高危预警！交由后端做终审判定...")
            # 【Stage 3】：极耗算力的循环功率谱最终确认
            time_s3 = time.time()
            confirm_flag, audit_score = stage3_audit.run_spectral_audit(stage1_scan.sdr)
            cost_s3 = time.time() - time_s3
            print(f"【S3 耗时】{cost_s3:.2f} s | 循环谱结论: {confirm_flag} (分数 {audit_score:.4f})")
            
            if confirm_flag:
                print(f"🎯 [最终告警]: 判定为非法入侵无人机！(频段: {active_center_freq/1e6} MHz)")
                # 👉 在此处可以加入与 主控 UDP 发送或者触发前端报警数据库写入的接口
            else:
                print(f"👻 [虚惊一场]: 循环谱分析表明大概率是一股极其活跃的普通路由 Wi-Fi 信号。")
                
        # 主动让出 CPU 控制权，可根据需要休眠
        time.sleep(0.1)

if __name__ == "__main__":
    main()
