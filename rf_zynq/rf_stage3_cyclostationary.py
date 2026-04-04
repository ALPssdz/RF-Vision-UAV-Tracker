import numpy as np
from datetime import datetime

class RF_Stage3_CycloAudit:
    """
    认知射频第三阶段：审问 (Audit)
    从高危时段中提取底层IQ数据流，使用循环谱特征 (Cyclostationarity) 来进行真假无人机分离。
    防误报（假阳性）最重要的一环：用于分辨 WiFi 宽带大流量 与 DJI 图传信号的本质差异。
    """
    def __init__(self, sample_rate=40e6):
        self.sample_rate = sample_rate
        
        # --- 核心降维侦测公式参数 ---
        # 完整的循环谱 CSD 矩阵 S(f, alpha) 算力耗散 = O(N^2 logN)
        # 降维“特例验证”：只取待排查目标协议的已知符号步进提取 alpha_target。
        # 例如：假设目标图传底层调制使用了某种特定的循环前缀或是 250kHz/500kHz 频带包络基波
        # Alpha频率对应公式: alpha_discrete = alpha_Hz / sample_rate * 2 * pi
        
        # 【物理真相】：802.11 a/g/n/ac Wi-Fi 的标准 OFDM 子载波间隔 (Subcarrier Spacing) 是严丝合缝的 312.5 kHz (20MHz宽/64)，这是它的基因！
        self.alpha_wifi = 312.5e3      
        # 未知无人机图传大概率不会使用公开碰瓷的 312.5，我们暂定一个隔离点特征
        self.alpha_drone1 = 500.0e3
        
    def _compute_alpha_slice(self, complex_iq, alpha_hz):
        """
        按照一维滑动降频算法计算特定切片上的协方差强度：
        R_x^alpha(0) = 1/M * sum_m ( x(m) * x*(m) * exp(-j * 2pi * alpha_hz * m / Fs) )
        """
        # 将局域网 SDR 原始 int16 数据化压为物理浮点数，免得溢出天际
        normalized_iq = complex_iq / 32768.0
        
        N = len(normalized_iq)
        # 生成时间乘子数组: 0, 1, 2, ..., N-1
        m_array = np.arange(N)
        # 生成基于特定循环频率 alpha 的移频复指数: exp(-j * 2pi * alpha * t)
        phase_shift = np.exp(-1j * 2.0 * np.pi * alpha_hz * m_array / self.sample_rate)
        
        # 计算瞬时功率 x(m)*x*(m) 即为幅度平方，再乘以对应的移频指数
        power_shifted = (np.abs(normalized_iq) ** 2) * phase_shift
        
        # 求平均特征模量作为此参数下的钻石点密度
        score = np.abs(np.mean(power_shifted))
        return score
        
    def run_spectral_audit(self, sdr_instance):
        """
        当 YOLO 在第二阶段报警时，拉起此进程深度抓取真实信号做数学验证。
        """
        print("[Audit] 视觉警报拉响！开始底层基元特征终审...")
        # 丢弃一条脏数据
        _ = sdr_instance.rx()
        
        # 获取用于物理层数学计算的长长一段串行IQ
        # 这里提取连续 4 帧拼接，使得数据长度达到约 65536 点，以保证足够细致的 alpha 分辨力
        iq_audit = np.concatenate([sdr_instance.rx() for _ in range(4)])
        
        # 执行“特指 Alpha” 降维扫描而不是全平面扫！
        score_wifi = self._compute_alpha_slice(iq_audit, self.alpha_wifi)
        score_drone = self._compute_alpha_slice(iq_audit, self.alpha_drone1)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 循环谱分量量化 -> [宽带底噪 (Wi-Fi): {score_wifi:.4f}] | [目标驻留特征 (UAV): {score_drone:.4f}]")
        
        # 分类器坚壁清野逻辑：如果无人机特征强度必须至少是 Wi-Fi 底噪特征的 5 倍，且自身具备宏观分量！
        if score_drone > score_wifi * 5.0 and score_drone > 0.0001:
            return True, score_drone
        else:
            return False, score_wifi
            
if __name__ == "__main__":
    print("循环谱特征解析引擎已就绪。")
