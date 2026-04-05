"""
RF Stage 1: RSSI 快速功率扫描 (Fast RSSI Pre-Scan)
===================================================
在执行耗时的 S2 瀑布图绘制和 S3 循环谱审计之前，
先用极短的小缓冲区对所有扫描扇区进行快速能量测量，
找出 RSSI 最强的扇区，优先将 S2+S3 的算力集中在那里。

物理原理：
  RSSI = E[|x(t)|²]，即 IQ 样本功率均值，反映该扇区内的总辐射能量密度。
  当无人机出现在某一扇区时，其发射功率会使该扇区 RSSI 明显抬升。

优势：
  - 小缓冲区（16384 样本 = 0.4ms）切换速度极快，全三扇区扫完仅需 ~30ms
  - 减少在无信号扇区的无效 S2/S3 计算，系统响应速度提升约 2~3 倍
  - 不受 OcuSync 跳频影响（只要有发射功率即可检测）
"""

import numpy as np
import time


# 每个扇区用于 RSSI 估计的采样点数（6.5ms @ 40MSPS）
# 样本量为原来的 16×，功率均值统计更稳定，可有效抑制 Wi-Fi 突发包干扰
S1_BUFFER_SIZE = 262144

# S1 预扫的 RSSI 主导判定比值
# 最强扇区功率 ≥ 次强扇区的 1.5 倍，才视为"明确主导"
RSSI_DOMINANCE_RATIO = 1.5

# AD9364 在 5.8GHz 频段的 PLL 重新锁定等待时间
# 保守取 50ms，确保大频率跳变（如 5745→5825MHz 跨 80MHz）后 LO 完全稳定
PLL_SETTLE_MS = 0.050

# 每个扇区正式测量帧数（discard=1 + measure=N，取 N 帧均值）
# 双帧平均可将偶发 Wi-Fi 突发包对 RSSI 的影响降低至 1/N
S1_MEASURE_FRAMES = 2

# S2 主缓冲大小（S1 扫描结束后需还原）
S2_BUFFER_SIZE = 2621440


class RF_Stage1_RSSIScan:
    """
    快速 RSSI 预扫模块

    职责：
      1. 在 scan_and_rank() 开始时将缓冲区切至 S1_BUFFER_SIZE（只切一次）
      2. 依次调谐各扇区，PLL 稳定后先丢弃一帧（flush ADC 管道），再读一帧测量
      3. 按功率均值降序返回扇区列表
      4. 结束时将缓冲区还原至 S2_BUFFER_SIZE
    """

    def __init__(self, sdr, sweep_sectors: list, sample_rate: int = int(40e6)):
        self.sdr = sdr
        self.sectors = sweep_sectors
        self.sample_rate = sample_rate
        # EMA 历史（抑制偶发脉冲噪声对排序的干扰）
        self._rssi_smooth = {freq: 0.0 for freq in sweep_sectors}
        self._smooth_alpha = 0.5   # 适当提高响应速度

    # ------------------------------------------------------------------
    def _measure_rssi_at(self, freq_hz: float) -> float:
        """
        调谐至指定频率并测量 RSSI（功率均值）。

        RSSI 计算公式：
            P = (1/N) · Σ|x_i|²   （归一化线性功率）

        包含三步净化流程：
          ① LO 调谐 + PLL 稳定等待（20ms）
          ② rx_destroy_buffer() + 一帧丢弃读（flush ADC 管道残余）
          ③ 正式读一帧 → 去 DC 偏置 → 计算功率
        """
        # ① LO 调谐，等待 PLL 锁定（50ms：覆盖 AD9364 最大跨频段稳定时间）
        self.sdr.rx_lo = int(freq_hz)
        time.sleep(PLL_SETTLE_MS)

        # ② 清空 USB/DMA 管道（消除前一扇区的残余数据）
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        
        # ③ 丢弃一帧（flush：ADC pipeline 里还可能有 PLL 收敛前的"过渡帧"）
        try:
            _ = self.sdr.rx()
        except Exception:
            return 0.0

        # ④ 正式采集 S1_MEASURE_FRAMES 帧，取功率均值（抑制 Wi-Fi 突发包干扰）
        #    P_avg = (1/N) · Σ_n [ (1/L) · Σ_l |x_{n,l}|² ]
        powers = []
        for _ in range(S1_MEASURE_FRAMES):
            try:
                raw = self.sdr.rx()
                iq = raw.astype(np.float32) / 32768.0
                iq -= np.mean(iq)   # 去除 LO leakage 产生的 DC 偏置
                powers.append(float(np.mean(np.abs(iq) ** 2)))
            except Exception:
                pass
        
        return float(np.mean(powers)) if powers else 0.0

    # ------------------------------------------------------------------
    def scan_and_rank(self) -> list:
        """
        对所有扇区执行快速 RSSI 扫描，返回按功率降序排列的扇区列表。

        缓冲区管理策略：
          - 扫描开始前将 rx_buffer_size 切至 S1_BUFFER_SIZE（只切一次）
          - 扫描结束后还原至 S2_BUFFER_SIZE（只切一次）
          避免在每个扇区间反复切换缓冲区导致 DMA 管道污染。

        Returns
        -------
        list of (freq_hz, rssi_ema) : 按 RSSI 从高到低排列
        """
        # ── 步骤 1：切一次小缓冲（S1 专用） ────────────────────────────
        self.sdr.rx_buffer_size = S1_BUFFER_SIZE

        rssi_map = {}
        for freq in self.sectors:
            rssi_raw = self._measure_rssi_at(freq)
            # EMA 平滑：加权平均历史值，抑制单次脉冲噪声
            self._rssi_smooth[freq] = (
                self._smooth_alpha * rssi_raw
                + (1 - self._smooth_alpha) * self._rssi_smooth[freq]
            )
            rssi_map[freq] = self._rssi_smooth[freq]

        # ── 步骤 2：还原大缓冲（供 S2 使用） ────────────────────────────
        self.sdr.rx_buffer_size = S2_BUFFER_SIZE
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass

        # 降序排列
        ranked = sorted(rssi_map.items(), key=lambda kv: kv[1], reverse=True)

        # 打印扫描结果（格式：[S1] RSSI Pre-scan 结果）
        result_str = " | ".join(
            f"{f/1e6:.0f} MHz: {p*1e6:.2f} μW"
            for f, p in ranked
        )
        top_freq, top_rssi = ranked[0]
        second_rssi = ranked[1][1] if len(ranked) > 1 else top_rssi
        ratio = top_rssi / (second_rssi + 1e-12)
        dominant = ratio >= RSSI_DOMINANCE_RATIO
        status = f"dominant (ratio = {ratio:.2f}×)" if dominant else f"marginal (ratio = {ratio:.2f}×, EMA converging)"
        print(f"  [S1] RSSI Pre-scan: {result_str} → Priority sector: {top_freq/1e6:.0f} MHz [{status}]")
        return ranked
