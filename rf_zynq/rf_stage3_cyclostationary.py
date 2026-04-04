import numpy as np

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
        
        # 为了演示，此处添加两组典型特异指纹 Alpha
        self.alpha_wifi = 250e3      # 取 250kHz OFDM 典型子带
        self.alpha_drone1 = 500e3    # 取 500kHz 未知私有图传假定特征
        
    def _compute_alpha_slice(self, complex_iq, alpha_hz):
        """
        按照一维滑动降频算法计算特定切片上的协方差强度：
        R_x^alpha(0) = 1/M * sum_m ( x(m) * x*(m) * exp(-j * 2pi * alpha_hz * m / Fs) )
        """
        N = len(complex_iq)
        # 生成时间乘子数组: 0, 1, 2, ..., N-1
        m_array = np.arange(N)
        # 生成基于特定循环频率 alpha 的移频复指数: exp(-j * 2pi * alpha * t)
        phase_shift = np.exp(-1j * 2.0 * np.pi * alpha_hz * m_array / self.sample_rate)
        
        # 计算瞬时功率 x(m)*x*(m) 即为幅度平方，再乘以对应的移频指数
        power_shifted = (np.abs(complex_iq) ** 2) * phase_shift
        
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
        
        print(f"👉 循环谱提取得分：[WiFi指纹: {score_wifi:.4f}] | [Drone指纹: {score_drone:.4f}]")
        
        # 分类器逻辑：如果图传特异点的相关强度彻底压制了 WiFi 常模
        if score_drone > score_wifi * 1.5 and score_drone > 0.001:
            return True, score_drone
        else:
            return False, score_wifi
            
if __name__ == "__main__":
    print("循环谱特征解析引擎已就绪。")
