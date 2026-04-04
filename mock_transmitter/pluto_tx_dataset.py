import adi
import numpy as np
import time
import os
from scipy.signal import resample_poly

def main():
    print("="*60)
    print("RF-Vision 靶机模拟器: 数据集真实电磁波物理重放 (Dataset Replay)")
    print("="*60)
    
    # === 步骤 1：定位并提取高精度原始数据集 ===
    # 我们使用 DJI MINI4 PRO - VTSBW=20 (20MHz带宽) 的大吞吐量模式物理截获帧 (pack2_0-1s.iq)
    dataset_path = r"e:\Myprojects\RF-Vision-UAV-Tracker\Drone RF Data\DJI MINI4 PRO\VTSBW=20\pack2_0-1s.iq"
    
    if not os.path.exists(dataset_path):
        print(f"[!] 致命异常：找不到数据集基底文件，路径检查失败！\n-> {dataset_path}")
        return
        
    print("[1] 正在从高速硬盘挂载并解算底层二进制 IQ 波形矩阵...")
    # USRP 录制的 Complex Float 格式: 32位实部与32位虚部交错
    # 为了防止吃爆内存（800MB完整读取和下采样对测试是不必要的），
    # 我们只加载前 2,500,000 个采样位点（相当于原始时间的 25 毫秒，包含数十个完整通信帧列，足够撑起循环谱核验所需的数据域）
    frames_to_read = 2500000 
    
    # 按照 np.complex64 (两路 float32) 按块读取
    raw_iq = np.fromfile(dataset_path, dtype=np.complex64, count=frames_to_read)
    
    print(f"    成功提取 {len(raw_iq)} 个高维复平面节点 (对应原生带宽下的 25ms 物理态).")
    
    # === 步骤 2：对阵列实施数学级降维采样 (Down-sampling) ===
    # 数据集是 USRP 采集的 100 MSPS。而我们的靶机设定是 40 MSPS 采样发送。
    # 如果不做基带重采样，直接推出去的图传带会“紧缩”，导致底层物理 Cyclic Feature 漂移变形！
    print("[2] 正在启用 Polyphase FIR 算子，跨系重采样将 100MSPS 映射下降维至 40MSPS...")
    # up=2, down=5 -> 确保频率特性的绝对缩放锚定
    resampled_iq = resample_poly(raw_iq, up=2, down=5) 
    print(f"    硬件时域重对齐完毕，总计算量点减免至 {len(resampled_iq)} 采样节.")
    
    # === 步骤 3：数据格式定点浮雕转换与重组 ===
    # PlutoSDR 发射器强制吸收 [-32768, 32767] 大数字定点，必须做放大投射
    max_amp = np.max(np.abs(resampled_iq))
    if max_amp == 0: max_amp = 1.0 # 避免除零
    
    target_scale = 20000.0 # 避开硬件顶峰削波截断失真
    normalized_iq = (resampled_iq / max_amp) * target_scale
    
    # === 步骤 4：硬件节点网络建立并推入循环 FPGA 缓冲 ===
    uri = "ip:192.168.31.20"
    print(f"\n[3] 正在挂载网络物理执行端点 {uri} ...")
    try:
        sdr = adi.Pluto(uri)
    except Exception as e:
        print(f"[!] SDR 连接超时崩溃: {e}")
        return
        
    # 定义发射管线约束
    sdr.sample_rate = int(40e6)
    sdr.tx_rf_bandwidth = int(40e6)
    sdr.tx_lo = int(2450e6)          # 恢复原始物理特征截获频点 (2.45 GHz)
    sdr.tx_hardwaregain_chan0 = -5   # 加大火力，以压制环境底噪凸显数据集本体能量
    
    # ★ 使用 FPGA 级硬件内存闭环，不断重复这段图传录音
    sdr.tx_cyclic_buffer = True
    
    print(f"\n[!] 物理信令发射准备就绪。靶机正以 {sdr.sample_rate/1e6}MSPS 发射真实的 DJI MAVIC 3 PRO 物理级录制频段！")
    print(f"    射频驻留点锁定: {sdr.tx_lo/1e6} MHz")
    print(">>> 正在不间断辐射图传载波，您随时可以按 Ctrl+C 安全阻断并重置靶机。")
    
    # PUSH ！！
    sdr.tx(normalized_iq)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] 用户硬终端接收，正在抹除 FPGA 缓冲死区...")
        sdr.tx_cyclic_buffer = False
        sdr.tx(np.zeros(1024))
        try:
            sdr.tx_destroy_buffer()
        except:
            pass
        print("硬件映射清除完成。")

if __name__ == "__main__":
    main()
