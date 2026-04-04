import numpy as np
from datetime import datetime

class RF_Stage3_CycloAudit:
    """
    Cognitive RF Tier 3: Final Feature Audit Evaluation.
    获取所锁定的待定识别信道段区域内的基本元IQ复矩阵时序矢量层序列，
    进而提取运行带有显著基带参考循环谱属性分析特化协议处理分类模型。
    以物理特征原理层面强拒抗大量存在基于时分正交子频域 802.11 设备等引发的一型判定错误伪报现象 (False Positive 抑制)。
    """
    def __init__(self, sample_rate=40e6):
        self.sample_rate = sample_rate
        self.alpha_drone1 = 500.0e3
        
        # 【算法纠错升级】：DJI OcuSync 使用的是标准的物理层 OFDM 架构。
        # 样本延迟常数计算 (Fs = 40MHz)：
        # Wi-Fi 4/5/6 (802.11g/n/ac): 子载波 312.5kHz => T_u = 3.2微秒 => 延迟样本数 = 128
        # OcuSync 15kHz 子载波: 符号域长度 = 1/15kHz = 66.67微秒 => 延迟样本数 = 2667
        # OcuSync 30kHz 子载波: 符号域长度 = 1/30kHz = 33.33微秒 => 延迟样本数 = 1333
        self.delay_wifi_cp = 128
        self.delay_ocusync_15k = 2667
        self.delay_ocusync_30k = 1333

    def _compute_alpha_slice(self, complex_iq, alpha_hz):
        # 针对老式调幅遥控图传适用的零延迟包络谱积分
        normalized_iq = complex_iq / 32768.0
        N = len(normalized_iq)
        m_array = np.arange(N)
        phase_shift = np.exp(-1j * 2.0 * np.pi * alpha_hz * m_array / self.sample_rate)
        power_shifted = (np.abs(normalized_iq) ** 2) * phase_shift
        cyclic_amplitude = np.abs(np.mean(power_shifted))
        total_power = np.mean(np.abs(normalized_iq) ** 2) + 1e-12
        return cyclic_amplitude / total_power
        
    def _compute_cp_correlation(self, complex_iq, delay_samples):
        """
        【真理方程】：OFDM 循环协议克星！(基于时移自相关 Delayed Autocorrelation)
        提取任何伪装在白噪声下的无人机 OcuSync 基带特征。
        """
        normalized_iq = complex_iq / 32768.0
        
        # 扣除物理硬件级直流失调(DC Offset)与本振泄漏(LO Leakage)
        normalized_iq = normalized_iq - np.mean(normalized_iq)
        
        if len(normalized_iq) <= delay_samples: return 0.0
            
        iq_main = normalized_iq[delay_samples:]
        iq_delayed = normalized_iq[:-delay_samples]
        
        correlation = np.abs(np.mean(iq_main * np.conj(iq_delayed)))
        power_main = np.mean(np.abs(iq_main) ** 2) + 1e-12
        return correlation / power_main
        
    def run_spectral_audit(self, sdr_instance):
        _ = sdr_instance.rx()
        
        # 【大数定律压制方差】
        # 将取样区间暴增至 12 帧，根据高斯分布方差缩小原则，环境底噪将被熨平压死在极其稳定的极低基线上！
        iq_audit = np.concatenate([sdr_instance.rx() for _ in range(12)])
        
        score_drone = self._compute_alpha_slice(iq_audit, self.alpha_drone1)
        score_wifi_cp = self._compute_cp_correlation(iq_audit, self.delay_wifi_cp)
        score_cp_15k = self._compute_cp_correlation(iq_audit, self.delay_ocusync_15k)
        score_cp_30k = self._compute_cp_correlation(iq_audit, self.delay_ocusync_30k)
        
        print(f"      >> [S3 矩阵校验] Wi-Fi包({score_wifi_cp*100:.1f}%) | 传统AM({score_drone*100:.1f}%) | O4(15k)={score_cp_15k*100:.2f}% | O4(30k)={score_cp_30k*100:.2f}%")
        
        # =========================================================================
        # 【物理正交双峰定律】
        # 真正的 DJI 图传因为带着 OFDM 数据负载，当延迟到达两倍（15k 对应的 2667 步）时，
        # 无人机在这个跨度已经是完全随机的两个符号，相关性必然彻底崩塌！
        # 如果 15k 和 30k 都在极高位共振，说明这只是一个无意义的长连续单频环境干扰！
        # =========================================================================
        
        # O4宽频(30k) 必须作为主物理矛点突破动态防爆阈值
        dynamic_th_30k = max(0.060, score_wifi_cp * 0.35)
        
        # 并发极值衰减定律：15k 的残留波必须小于 30k 一半以上，否则视为连续干扰谐波
        is_true_ocusync = score_cp_30k > dynamic_th_30k and score_cp_15k < (score_cp_30k * 0.55)
        
        if is_true_ocusync:
            return True, score_cp_30k
            
        elif score_drone > 0.055:
            return True, score_drone
            
        elif score_wifi_cp > 0.040:
            print(f"      >> [S3 诊断报告] 协议拦截！命中极强 IEEE 802.11 OFDM 背景包裹。")
            return False, score_wifi_cp
            
        else:
            return False, score_wifi_cp
