import adi
import numpy as np
import time

class RF_Stage1_Sweeper:
    """
    认知射频第一阶段：扫描（Scan）
    将2.4G宽带(80MHz)分成两段，以极快速度轮巡比对能量 (RSSI)。
    """
    def __init__(self, uri="ip:192.168.31.10"):
        # 参数初始化
        self.sample_rate = int(40e6)
        self.buffer_size = 16384
        
        # 定义双子信道频率中心
        self.freq_segment_a = int(2420e6) # 涵盖 2400-2440MHz
        self.freq_segment_b = int(2460e6) # 涵盖 2440-2480MHz
        
        self.uri = uri
        self.sdr = None

    def initialize_sdr(self):
        try:
            self.sdr = adi.Pluto(self.uri)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.rx_rf_bandwidth = self.sample_rate
            self.sdr.rx_buffer_size = self.buffer_size
            # AGC 在跳频时可能会出现延迟适应，在特定环境建议改成 'manual' 然后配合固定高增益
            self.sdr.rx_hardwaregain_control_mode = 'slow_attack' 
            print("✅ Stage1 Sweeper SDR Initialized.")
        except Exception as e:
            print(f"❌ 无法连接到 SDR 设备: {e}")
            raise e

    def compute_rssi_db(self, iq_data):
        """
        计算这段时域复数流的能量大小 (公式规范)
        P_avg = sum(|I + jQ|^2) / N
        RSSI = 10 * log10(P_avg)
        """
        # 计算每一位复数点对应的绝对功率, 再算平均
        power_linear = np.mean(np.abs(iq_data)**2)
        # 加入 1e-12 避免对0取对数崩溃
        rssi_db = 10 * np.log10(power_linear + 1e-12)
        return rssi_db

    def run_sweep_cycle(self):
        """
        执行一次“双跳”巡视：获取A/B两段的能量比对。
        返回建议凝视(Dwell)的中心频率。
        """
        # ==========================================
        # 1. 切向段 A 获取 RSSI
        # ==========================================
        self.sdr.rx_lo = self.freq_segment_a
        # 由于写入底层寄存器存在硬件锁相所需时间(t_tune约30ms)，在切频后扔掉前两包残存的缓冲
        _ = self.sdr.rx()
        _ = self.sdr.rx()
        data_a = self.sdr.rx()
        rssi_a = self.compute_rssi_db(data_a)
        
        # ==========================================
        # 2. 切向段 B 获取 RSSI
        # ==========================================
        self.sdr.rx_lo = self.freq_segment_b
        _ = self.sdr.rx()
        _ = self.sdr.rx()
        data_b = self.sdr.rx()
        rssi_b = self.compute_rssi_db(data_b)
        
        # ==========================================
        # 3. 能量判定
        # ==========================================
        print(f"[Sweep] Freq A (2.42G): {rssi_a:.2f} dB  | Freq B (2.46G): {rssi_b:.2f} dB")
        
        # 返回能量更高的那一段中心频率，以便主控指导 Stage2 Dwell
        # （可选扩展：可以引入最低环境底噪门限如 40dB，低于门限表示绝对风平浪静，完全不调用模型）
        return self.freq_segment_a if rssi_a > rssi_b else self.freq_segment_b

if __name__ == "__main__":
    sweeper = RF_Stage1_Sweeper()
    sweeper.initialize_sdr()
    
    print("启动信道扫描巡视模式...")
    # 测试跑10回合扫描测算物理时间
    start = time.time()
    for i in range(10):
        target_freq = sweeper.run_sweep_cycle()
        print(f"回合 {i+1} 判定优胜驻留频点 -> {target_freq / 1e6} MHz")
    cost = time.time() - start
    print(f"10次全频段轮回跳频总耗时: {cost:.2f} 秒 (平均单跳单段耗时约为 {(cost/20)*1000:.1f} ms)")
    
    import sys
    sys.exit(0)
