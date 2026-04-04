from ultralytics import YOLO
import os

def train_rf_model():
    """
    针对 RFUAV 提取数据集的训练脚本。
    执行位置在 rf_zynq/yolo 下，已修复指向目标 yaml 的绝对路径
    """
    print("🚀 准备加载 YOLOv8 Nano 预训练大模型...")
    model = YOLO('yolov8n.pt')  

    # 指向根目录生成的 yaml
    yaml_config = "e:/Myprojects/RF-Vision-UAV-Tracker/yolo_dataset/rf_uav.yaml"

    print("🧠 开始在自定义特征库上狂飙学习...")
    results = model.train(
        data=yaml_config,  
        epochs=150,          
        imgsz=640,           
        batch=16,           
        device=0             
    )
    
    print("\n🎉 训练圆满完成！")

if __name__ == '__main__':
    train_rf_model()
