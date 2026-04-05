import numpy as np
from datetime import datetime

class RF_Stage3_CycloAudit:
    """
    Stage 3: OcuSync Cyclostationary Spectral Audit Module

    基于循环平稳信号分析理论（Cyclostationary Signal Analysis），
    对 Stage 2 采集的原始 IQ 数据进行 OFDM 循环前缀（Cyclic Prefix, CP）
    时移自相关检测，以提取 DJI OcuSync 协议的物理层周期性特征。

    检测原理：
      OFDM 信号具有循环平稳性。其循环前缀长度为 T_cp，对应延迟 Tau = T_cp × Fs 个样本。
      时移自相关函数 R(τ) = E[x*(t) · x(t+τ)] 在 τ = T_cp 处呈现显著峰值。
      归一化后的峰值强度反映了该延迟对应协议的能量占比，可用于协议指纹识别。

    目标延迟参数（Fs = 40 MSps）：
      - IEEE 802.11 (Wi-Fi): 子载波间隔 Δf = 312.5 kHz → T_u = 3.2 μs → τ_wifi = 128 samples
      - OcuSync 2.0 (15 kHz): Δf = 15 kHz → T_u = 66.67 μs → τ_15k = 2667 samples
      - OcuSync 3.0/4.0 (30 kHz): Δf = 30 kHz → T_u = 33.33 μs → τ_30k = 1333 samples

    虚警抑制机制：
      1. 功率门控：接收功率低于 MIN_POWER_GATE 时跳过（防止低 SNR 下归一化失真）
      2. 双通道独立判决：τ_30k 和 τ_15k 各自独立与对应阈值比较，避免通道间阈值混用
      3. 电源纹波鉴别：真实 OcuSync CP 峰在时延域呈 Delta 冲激，旁开 5 samples 即衰减至噪声底；
         DC-DC SMPS 纹波为低频连续正弦，旁开后仍保持高相关性，据此区分两者
      4. Wi-Fi 自适应阈值：当 τ_128 处得分较高时，动态上调检测阈值，防止 Wi-Fi 溢出误触发
    """

    def __init__(self, sample_rate=40e6):
        self.sample_rate = sample_rate

        # 各协议的循环前缀时延样本数（基于 OFDM 子载波间隔计算）
        # τ = round(Fs / Δf)，其中 Δf 为子载波间隔频率
        self.delay_wifi_cp     = 128   # IEEE 802.11 a/g/n/ac, Δf = 312.5 kHz
        self.delay_ocusync_15k = 2667  # OcuSync 2.0 (DJI Mini 3/Air 2S), Δf = 15 kHz
        self.delay_ocusync_30k = 1333  # OcuSync 3.0/4.0 (DJI Mini 4 Pro/Mavic 3), Δf = 30 kHz

    # =========================================================================
    # 检测阈值（基于现场环境实测背景数据校准，v3.0）
    #
    # 测试环境：室内 5.8GHz ISM 频段，无人机关机状态，采集 3 组背景数据取最大值：
    #   τ=1333 背景峰值：3.29%（5745 MHz 扇区）
    #   τ=2667 背景峰值：2.98%（5825 MHz 扇区）
    #
    # 阈值设定准则：γ = β_bg_max × α_margin
    #   其中 α_margin = 2.0（安全余量因子，确保虚警概率 Pfa ≪ 1 同时保留足够检测余量）
    #   THRESHOLD_30K = 3.29% × 2.0 = 6.6% → 取 7%（工程上调整）
    #   THRESHOLD_15K = 2.98% × 2.0 = 5.96% → 取 6%
    # =========================================================================
    THRESHOLD_30K = 0.07  # τ=1333 通道检测阈值（OcuSync 30 kHz 子载波，Mini 4 Pro）
    THRESHOLD_15K = 0.06  # τ=2667 通道检测阈值（OcuSync 15 kHz 子载波，Mini 3）

    # 最低可信信号功率门控阈值（归一化 IQ 功率）
    # 依据：5.8 GHz 相较 2.4 GHz 自由空间路径损耗增量：
    #   ΔL = 20·log₁₀(5800/2450) = +7.5 dB，即接收功率下降约 5.6 倍
    # 实测 5785 MHz 扇区背景功率约 6.6×10⁻⁵，将门控阈值设为 1×10⁻⁵
    # 低于此功率时，SNR 不足以支撑可信的归一化相关估计，直接返回 0
    MIN_POWER_GATE = 1e-5

    # -------------------------------------------------------------------------
    def _compute_cp_correlation(self, complex_iq, delay_samples):
        """
        计算归一化时移自相关系数（Normalized Delayed Autocorrelation）。

        定义：
            R(τ) = |E[x*(t) · x(t+τ)]| / E[|x(t+τ)|²]

        其中 τ = delay_samples，x(t) 为基带复包络 IQ 序列。
        对 OFDM 信号，R(τ = T_cp·Fs) 处出现显著峰值，
        峰值幅度反映循环前缀能量占总信号功率的比例。

        Returns
        -------
        float : 归一化自相关系数（0~1），0 表示无周期性特征或功率不足
        """
        # 归一化至 [-1, 1] 量程，去除 ADC 整数量化偏置
        normalized_iq = complex_iq / 32768.0

        # 去除直流失调（DC Offset）与本振泄漏（LO Leakage）
        normalized_iq = normalized_iq - np.mean(normalized_iq)

        if len(normalized_iq) <= delay_samples:
            return 0.0

        iq_main    = normalized_iq[delay_samples:]
        iq_delayed = normalized_iq[:-delay_samples]

        power_main = np.mean(np.abs(iq_main) ** 2)

        # 功率门控：接收功率低于阈值时，归一化结果由热噪声涨落主导，不可信
        if power_main < self.MIN_POWER_GATE:
            return 0.0

        # 计算复互相关并归一化
        correlation = np.abs(np.mean(iq_main * np.conj(iq_delayed)))
        return correlation / (power_main + 1e-12)

    # -------------------------------------------------------------------------
    def run_spectral_audit(self, iq_data_buffer):
        """
        对输入 IQ 缓冲区执行完整的循环谱审计流程。

        算法流程：
          1. 滑动窗口分帧（窗口长度 200 000 采样点，50% 重叠）
          2. 对每帧并行计算 τ_30k 和 τ_15k 的归一化自相关系数
          3. 记录各通道在全缓冲区内的峰值（peak_30k, peak_15k）
          4. 分别与对应阈值（动态调整后）进行独立比较，任一超标进入验证流程
          5. 旁开点验证：计算 τ-5 处的相关值，判断时延峰的尖锐度
             - 峰值尖锐（adj/peak < 0.6）：与 OFDM CP 特征一致 → 判定为 OcuSync
             - 峰值宽平（adj/peak ≥ 0.6）：与连续波纹波特征一致 → 判定为 SMPS 干扰

        Parameters
        ----------
        iq_data_buffer : array-like
            来自 AD9364 ADC 的原始整数 IQ 数据（int16 格式，I/Q 交织或复数数组）

        Returns
        -------
        (bool, float) : (检测结果, 最大相关系数)
            True 表示检测到 OcuSync 协议特征，False 表示未检测到或被虚警抑制
        """
        chunk_size   = 200000
        step_size    = chunk_size // 2  # 50% 重叠，保证帧边界处的突发包不被漏检
        total_samples = len(iq_data_buffer)

        # 第一遍扫描：追踪全局最大值（用于确定最佳候选帧及 Wi-Fi 伴随指标）
        max_30k      = 0.0
        assoc_wifi_cp = 0.0
        best_chunk   = None
        target_delay = self.delay_ocusync_30k

        for i in range(0, total_samples - chunk_size, step_size):
            chunk = iq_data_buffer[i : i + chunk_size]
            score_cp_30k = self._compute_cp_correlation(chunk, self.delay_ocusync_30k)
            score_cp_15k = self._compute_cp_correlation(chunk, self.delay_ocusync_15k)
            local_max = max(score_cp_30k, score_cp_15k)

            # 记录含最高 OcuSync 特征的候选帧，同步提取其 Wi-Fi 伴随相关系数
            if local_max > max_30k:
                max_30k       = local_max
                assoc_wifi_cp = self._compute_cp_correlation(chunk, self.delay_wifi_cp)
                best_chunk    = chunk
                target_delay  = self.delay_ocusync_30k if score_cp_30k > score_cp_15k else self.delay_ocusync_15k

        print(f"  [S3] Background correlation — R(τ=128)={assoc_wifi_cp*100:.1f}% | OcuSync max peak: {max_30k*100:.2f}%")

        # =====================================================================
        # 双通道独立判决（Independent Dual-Channel Decision）
        #
        # 设计依据：τ_30k 与 τ_15k 分别对应不同型号无人机的协议规格，
        # 信号强度存在差异，不应共用同一阈值进行首级门限判断。
        # 采用各通道独立与其对应阈值比较，OR 逻辑触发后验证，可提升检测概率。
        # =====================================================================
        th_30k = self.THRESHOLD_30K
        th_15k = self.THRESHOLD_15K

        # Wi-Fi 自适应阈值调整：
        # 当 τ_128 处相关系数较高时，表明环境中存在强 Wi-Fi 干扰，
        # 动态上调阈值以抑制 Wi-Fi 频谱溢出（spectral spillover）导致的虚警
        dynamic_th_30k = max(th_30k, assoc_wifi_cp * 0.40)
        dynamic_th_15k = max(th_15k, assoc_wifi_cp * 0.40)

        # 第二遍扫描：各通道独立追踪峰值（避免通道间最大值互相掩盖）
        peak_30k = 0.0
        peak_15k = 0.0
        for i in range(0, total_samples - chunk_size, step_size):
            chunk = iq_data_buffer[i : i + chunk_size]
            s30 = self._compute_cp_correlation(chunk, self.delay_ocusync_30k)
            s15 = self._compute_cp_correlation(chunk, self.delay_ocusync_15k)
            if s30 > peak_30k:
                peak_30k     = s30
                target_delay = self.delay_ocusync_30k
            if s15 > peak_15k:
                peak_15k     = s15
                best_chunk   = chunk
                target_delay = self.delay_ocusync_15k

        print(f"  [S3] Dual-channel peaks — τ=1333: {peak_30k*100:.2f}% (th={dynamic_th_30k*100:.1f}%) | "
              f"τ=2667: {peak_15k*100:.2f}% (th={dynamic_th_15k*100:.1f}%)")

        # OR 逻辑：任一通道超过其对应阈值，即进入验证阶段
        triggered       = False
        triggered_score = 0.0
        if peak_30k > dynamic_th_30k:
            triggered       = True
            triggered_score = max(triggered_score, peak_30k)
        if peak_15k > dynamic_th_15k:
            triggered       = True
            triggered_score = max(triggered_score, peak_15k)

        if triggered:
            # =================================================================
            # 时延旁瓣验证（Adjacent-Tap Sharpness Test）
            #
            # 原理：真实 OFDM 循环前缀的时移自相关在 τ = T_cp 处呈现 Delta 函数特征，
            # 偏移 ±5 个采样点后，相关值应迅速衰减至背景噪声水平（< 1%）。
            # DC-DC SMPS 的开关纹波为低频连续正弦波，其自相关函数为正弦包络，
            # 旁开 5 点后相关值衰减有限（> 60% 峰值），可据此区分两种干扰源。
            # =================================================================
            if best_chunk is not None:
                adj_corr = self._compute_cp_correlation(best_chunk, target_delay - 5)
                if adj_corr > (triggered_score * 0.60):
                    print(f"  [S3] Adjacent-tap test FAILED: R(τ-5)={adj_corr*100:.1f}% > "
                          f"0.60 × peak({triggered_score*100:.1f}%) — SMPS ripple interference, alert suppressed.")
                    return False, triggered_score

                # 检测确认：生成循环谱诊断图像（用于事后分析）
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    import matplotlib.pyplot as plt
                    import os

                    print("  [S3] Generating cyclostationary spectrum snapshot...")
                    delays_scan = np.arange(100, 3000, 10)
                    corrs_scan  = [self._compute_cp_correlation(best_chunk, d) for d in delays_scan]

                    plt.figure(figsize=(10, 4))
                    plt.plot(delays_scan, corrs_scan, color='#FF5722', linewidth=1.2,
                             label='Normalized Autocorrelation')
                    plt.axvline(1333, color='#1565C0', linestyle='--',
                                label=f'τ=1333 (OcuSync 30kHz): {peak_30k*100:.1f}%')
                    plt.axvline(2667, color='#2E7D32', linestyle='--',
                                label=f'τ=2667 (OcuSync 15kHz): {peak_15k*100:.1f}%')
                    plt.axhline(dynamic_th_30k, color='#1565C0', linestyle=':', alpha=0.6,
                                label=f'Threshold 30k = {dynamic_th_30k*100:.1f}%')
                    plt.axhline(dynamic_th_15k, color='#2E7D32', linestyle=':', alpha=0.6,
                                label=f'Threshold 15k = {dynamic_th_15k*100:.1f}%')
                    plt.title(f"S3 Cyclostationary Audit — OcuSync Detected "
                              f"(τ₃₀: {peak_30k*100:.2f}% | τ₁₅: {peak_15k*100:.2f}%)")
                    plt.xlabel("Delay τ (samples)  [Fs = 40 MSps]")
                    plt.ylabel("Normalized Autocorrelation Coefficient")
                    plt.grid(alpha=0.3)
                    plt.legend(fontsize=8)
                    plt.tight_layout()

                    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "database", "alert_images")
                    os.makedirs(db_dir, exist_ok=True)
                    plt.savefig(os.path.join(db_dir, "S3_Cyclo_Snapshot.png"), dpi=120)
                    plt.close()
                    print("  [S3] Snapshot saved → database/alert_images/S3_Cyclo_Snapshot.png")
                except Exception as e:
                    print(f"  [S3] Snapshot generation error: {e}")

            return True, triggered_score

        # Wi-Fi 高强度拦截日志
        elif assoc_wifi_cp > 0.040:
            print(f"  [S3] Wi-Fi spillover suppression: R(τ=128)={assoc_wifi_cp*100:.1f}% > 4.0% "
                  f"— classified as IEEE 802.11 spectral spillover, alert suppressed.")
            return False, assoc_wifi_cp

        return False, max_30k
