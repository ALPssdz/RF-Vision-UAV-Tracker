import adi
import numpy as np
import time

class RF_Stage1_Sweeper:
    """
    Cognitive RF Tier 1: Wide-band Energy Spectrum Sweeping.
    将具备 80MHz 覆盖率的宽频带谱段物理划分成为隔离的运行子区域，
    以便执行连续波遍历轮巡以及接收机能量获取量度分析（RSSI）。
    """
    def __init__(self, uri="ip:192.168.31.10"):
        self.sample_rate = int(40e6)
        self.buffer_size = 16384
        
        # 定义双子信道频点的中心驻留常量参数
        self.freq_segment_a = int(2420e6) # 矢量段定义在 2400-2440MHz 区域
        self.freq_segment_b = int(2460e6) # 矢量段定义在 2440-2480MHz 区域
        
        self.uri = uri
        self.sdr = None

    def initialize_sdr(self):
        try:
            self.sdr = adi.Pluto(self.uri)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.rx_rf_bandwidth = self.sample_rate
            self.sdr.rx_buffer_size = self.buffer_size
            # 建立固化的 AGC 自动增益边界等效常数级参数（基于慢建立配置）
            # 改为手动底噪增益阈值，避免环境宽带底噪产生自动过激放大触发 YOLO 全屏假阳性
            self.sdr.rx_hardwaregain_control_mode = 'manual'
            self.sdr.rx_hardwaregain_chan0 = 20 
            print("[INFO] Stage1: SDR Initialization Protocol Completed.")
        except Exception as e:
            print(f"[ERROR] Stage1: Failed referencing the allocated SDR device socket address: {e}")
            raise e

    def compute_rssi_db(self, iq_data):
        """
        执行特定复平面空间内 IQ 数据矩阵包络的绝对射频能量获取以及算法标量计算。
        公式表达模型：P_avg = sum(|I + jQ|^2) / N, 随后转化成 RSSI 对数表示级： 10 * log10(P_avg)
        """
        power_linear = np.mean(np.abs(iq_data)**2)
        rssi_db = 10 * np.log10(power_linear + 1e-12)
        return rssi_db

    def run_sweep_cycle(self):
        """
        发起对双载波定义边界层的跳频评估运算调用。
        根据输出的比对能量返回系统认定能量激活量度占优的最佳中心频点参考系。
        """
        # ==========================================
        # 1. 区域 A 分带 RSSI 特征采集验证点
        # ==========================================
        self.sdr.rx_lo = self.freq_segment_a
        # 抛弃首发几个由于底层基于 PLL 硬件锁相耗时约 30ms 时间窗特性生成的不可靠包
        _ = self.sdr.rx()
        _ = self.sdr.rx()
        
        # 串行合并提取 20 个连续循环缓冲期信号点，进行时域滤波以抑制偶发性瞬态脉冲引入的不可靠量积
        data_a = np.concatenate([self.sdr.rx() for _ in range(20)])
        rssi_a = self.compute_rssi_db(data_a)
        
        # ==========================================
        # 2. 区域 B 分带 RSSI 特征采集验证点
        # ==========================================
        self.sdr.rx_lo = self.freq_segment_b
        _ = self.sdr.rx()
        _ = self.sdr.rx()
        
        data_b = np.concatenate([self.sdr.rx() for _ in range(20)])
        rssi_b = self.compute_rssi_db(data_b)
        
        # ==========================================
        # 3. 相对标量能量决断判定比较级
        # ==========================================
        return self.freq_segment_a if rssi_a > rssi_b else self.freq_segment_b

if __name__ == "__main__":
    sweeper = RF_Stage1_Sweeper()
    sweeper.initialize_sdr()
    
    start = time.time()
    for i in range(10):
        target_freq = sweeper.run_sweep_cycle()
    cost = time.time() - start
    
    import sys
    sys.exit(0)
