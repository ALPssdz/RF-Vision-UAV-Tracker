# -*- coding: utf-8 -*-
"""
rf_stage3_cyclostationary.py — Stage 3: Cyclic Frequency Discriminator (CFD v3.0)
===================================================================================

算法核心升级：从 α=0 标准自相关 → CAF-FFT 循环频率判别器
===========================================================

【旧算法局限】
  R_x^0(τ) = E[x(t) · x*(t-τ)]
  ↳ 仅在时延域区分协议，当 WiFi 功率 >> UAV 时，τ=128(WiFi) 处能量溢出到
    τ=1333/2667(OcuSync) 导致虚警，同时也掩盖微弱 OcuSync 信号。

【新算法原理：循环自相关函数 CAF】
  R_x^α(τ) = (1/N) Σ x[n]·x*[n-τ]·exp(-j2π α n/Fs)

  其中 α 称为"循环频率"，对应信号的周期性循环速率：
    · OcuSync 30kHz：  α₀ = Fs/(N_fft + N_cp) ≈ 24 kHz  （τ = 1333,  CP=1/4）
    · OcuSync 15kHz：  α₀ =  Fs/(N_fft + N_cp) ≈ 12 kHz  （τ = 2667,  CP=1/4）
    · IEEE 802.11 WiFi：α₀ = 1/T_sym = 250 kHz             （τ = 128）

  各协议循环频率完全分离。当以 OcuSync 的 α₀ 解调时，WiFi 的贡献为：
    NCC_WiFi(α_OcuSync, τ) ≈ A_WiFi · sinc((α_WiFi - α_OcuSync) · N / Fs)
                            ≈ A_WiFi · sinc(226kHz × 200000 / 40MHz)
                            ≈ A_WiFi · sinc(1130) ≈ 0.028%

  即使 WiFi 比无人机信号强 20 dB（10×），其在 OcuSync 循环频率通道上的泄漏
  仍为 0.028%，远低于 OcuSync 信号本身的 NCC（典型值 5~25%）。

【实现方式：FFT 加速 α 扫描】
  对于每帧、每个目标时延 τ，仅需：
    1. z[n] = x[n] · x*[n-τ]           # 时延乘积序列，O(N)
    2. Z[k] = FFT(z)                    # 一次 FFT 覆盖所有 α，O(N log N)
    3. NCC[α] = |Z[k]| / (N · P_x)     # 提取 OcuSync α-范围内最大值，O(1)

  相比逐点扫描大幅降低计算量，同时完整保留 α 域分辨率。

目标延迟（Fs = 40 MSps）：
  · τ=1333 → N_fft = Fs/30kHz → OcuSync 3.0/4.0 (Mini 4 Pro, Mavic 3)
  · τ=2667 → N_fft = Fs/15kHz → OcuSync 2.0     (Mini 3, Air 2S)

OcuSync α 扫描范围（覆盖 CP 比例 1/8 ~ 1/3 的全部情形）：
  · OcuSync 30kHz: α ∈ [18 kHz, 32 kHz]
  · OcuSync 15kHz: α ∈ [ 9 kHz, 16 kHz]
"""

import numpy as np
from datetime import datetime


class RF_Stage3_CycloAudit:
    """
    Stage 3 v3.0 — CAF-FFT 循环频率判别器

    公开接口与 v2.x 完全兼容：
        run_spectral_audit(iq_data_buffer) → (bool, float)
    """

    # ─── 协议物理参数 ────────────────────────────────────────────────────────
    TAU_OCUSYNC_30K = 1333   # OcuSync 3.0/4.0  N_fft = 40MHz/30kHz
    TAU_OCUSYNC_15K = 2667   # OcuSync 2.0       N_fft = 40MHz/15kHz
    TAU_WIFI        = 128    # IEEE 802.11        N_fft = 40MHz/312.5kHz

    # OcuSync 循环频率扫描范围（Hz），覆盖 CP_ratio ∈ [1/8, 1/3]
    # α_sym = Fs / (N_fft + N_cp) = Fs / (N_fft · (1 + CP_ratio))
    #   CP=1/8 → α = 40MHz/(1333·1.125) = 26.7 kHz (30k 通道)
    #   CP=1/3 → α = 40MHz/(1333·1.333) = 22.5 kHz
    ALPHA_SCAN_30K = (18_000.0, 32_000.0)   # 扫描窗口略宽，余量 ±2kHz
    ALPHA_SCAN_15K = ( 9_000.0, 16_000.0)

    # WiFi 监控循环频率（仅用于日志输出，不参与判决）
    ALPHA_WIFI_HZ  = 250_000.0

    # ─── 决策阈值 ────────────────────────────────────────────────────────────
    # CAF-NCC 噪声底理论值：≈ 1/√N = 1/√200000 ≈ 0.22%
    # 保守设定为噪声底的 10~12 倍（Pfa << 0.1%），同时兼顾弱信号灵敏度
    THRESHOLD_30K = 0.028   # 2.8%（原 7%；WiFi 已被循环频率正交性抑制）
    THRESHOLD_15K = 0.022   # 2.2%（原 6%）

    # 联合统计量权重：peak_weight × 峰值 + (1 - peak_weight) × 帧均值
    # 较高的均值权重使持续弱信号有更多帧叠加 SNR 增益
    PEAK_WEIGHT = 0.45

    # τ 域峰值旁瓣比（PSR）阈值
    # OFDM CP 峰为 Delta 冲激，PSR >> 1；SMPS/宽带干扰 PSR ≈ 1
    PSR_THRESHOLD      = 2.5   # 普通环境
    PSR_THRESHOLD_WIFI = 3.2   # 强 WiFi 监控区上调

    # α 域循环频率集中度（Cyclic Frequency Sharpness, CFS）阈值
    # 真实 OcuSync：α 峰尖锐（CFS > CFS_TH）；宽带干扰：α 峰平坦（CFS < CFS_TH）
    CFS_THRESHOLD = 1.8

    # 功率门控（与 v2.x 保持一致）
    MIN_POWER_GATE = 1e-5

    # ─── 帧分析参数 ──────────────────────────────────────────────────────────
    CHUNK_SIZE = 200_000
    OVERLAP    = 0.50          # 50% 帧重叠，确保帧边界处的突发包不被漏检

    def __init__(self, sample_rate: float = 40e6):
        self.sample_rate = float(sample_rate)
        self._freq_res   = None   # 将在首次 run_spectral_audit 按实际 chunk 设置

    # =========================================================================
    # 内核：CAF-FFT 计算
    # =========================================================================
    def _prepare_chunk(self, raw_chunk) -> tuple:
        """
        共用预处理：归一化 + DC 去除 + 功率计算。

        Returns
        -------
        (x, power) : x 为 complex64 基带序列，power 为归一化均值功率。
                     若功率不足（低 SNR），返回 (None, 0.0)。
        """
        x = raw_chunk.astype(np.complex64) / 32768.0
        x -= x.mean()
        power = float(np.mean(np.abs(x) ** 2))
        if power < self.MIN_POWER_GATE:
            return None, 0.0
        return x, power

    def _compute_caf_spectrum(self, x: np.ndarray, tau: int, power: float) -> np.ndarray:
        """
        计算归一化循环自相关谱（Normalized CAF Spectrum）。

        原理：
          z[n] = x[n] · x*[n-τ]  （时延 τ 的乘积序列）
          Z[k] = FFT(z)           （其频谱 = 循环自相关函数在 α 轴上的分布）
          NCC[α] = |Z[k]| / (N_z · P_x)

        当 α 等于信号真实循环频率 α₀ = 1/T_sym 时，NCC 取峰值（≈ CP 比例）；
        当 α ≠ α₀ 时，NCC ≈ 1/√N（噪声底）。

        Parameters
        ----------
        x     : 归一化 IQ 序列（已去 DC）
        tau   : 目标时延（样本数）
        power : x 的均值功率（用于归一化）

        Returns
        -------
        ncc_spectrum : 归一化 CAF 幅度谱，长度 = len(x) - tau，频率分辨率 = Fs/N
        """
        if len(x) <= tau:
            return np.zeros(1)

        z   = x[tau:] * np.conj(x[:-tau])
        N_z = len(z)
        Z   = np.fft.fft(z.astype(np.complex64))
        # 归一化：除以 N_z（窗口长度）和功率，得到无量纲相干系数
        return np.abs(Z) / (N_z * (power + 1e-12))

    def _extract_ncc_in_range(
        self,
        ncc_spectrum: np.ndarray,
        chunk_len: int,
        alpha_range_hz: tuple,
    ) -> tuple:
        """
        从 CAF 谱中提取指定 α 范围内的峰值。

        Parameters
        ----------
        ncc_spectrum   : _compute_caf_spectrum 返回的归一化谱
        chunk_len      : 原始 chunk 长度（用于 bin 计算）
        alpha_range_hz : (alpha_lo_hz, alpha_hi_hz)

        Returns
        -------
        (peak_ncc, best_alpha_hz, cfs)
          peak_ncc     : 范围内最大 NCC 值
          best_alpha_hz: 对应的最佳循环频率（Hz）
          cfs          : 循环频率集中度 = peak / median(其余 bin in range)
        """
        N     = len(ncc_spectrum)
        f_res = self.sample_rate / N          # Hz / bin

        k_lo = max(1, int(np.round(alpha_range_hz[0] / f_res)))
        k_hi = min(N // 2, int(np.round(alpha_range_hz[1] / f_res)) + 1)

        if k_lo >= k_hi:
            return 0.0, alpha_range_hz[0], 1.0

        segment    = ncc_spectrum[k_lo:k_hi]
        peak_idx   = int(np.argmax(segment))
        peak_ncc   = float(segment[peak_idx])
        best_alpha = (k_lo + peak_idx) * f_res

        # 循环频率集中度（CFS）：峰值 / 旁瓣中值
        sidelobes = np.delete(segment, peak_idx)
        cfs = peak_ncc / (float(np.median(sidelobes)) + 1e-12) if len(sidelobes) else 1.0

        return peak_ncc, best_alpha, cfs

    def _compute_psr(
        self,
        x: np.ndarray,
        power: float,
        tau_target: int,
        alpha_best_hz: float,
        delta_guard: int = 25,
        n_side: int = 10,
        half_w: int = 200,
    ) -> float:
        """
        τ 域峰值旁瓣比（PSR）。

        PSR(τ_0) = NCC(α_best, τ_0) / median{ NCC(α_best, τ) : |τ-τ_0| > δ_guard }

        真实 OFDM CP 峰（Delta 冲激型）：PSR >> 1
        SMPS 开关纹波 / 宽带干扰（连续型）：PSR ≈ 1

        使用已知最佳 α 在旁瓣 τ 点上重新计算单点 CAF，无需再做 FFT。
        """
        def single_caf(tau_probe: int) -> float:
            if len(x) <= tau_probe:
                return 0.0
            z  = x[tau_probe:] * np.conj(x[:-tau_probe])
            pw = float(np.mean(np.abs(x[tau_probe:]) ** 2))
            if pw < self.MIN_POWER_GATE:
                return 0.0
            n      = np.arange(len(z), dtype=np.float32)
            demod  = np.exp(-1j * 2.0 * np.pi * alpha_best_hz / self.sample_rate * n)
            return float(np.abs(np.mean(z * demod))) / (pw + 1e-12)

        peak = single_caf(tau_target)

        # 构造旁瓣采样点（保护带外均匀分布）
        lo = max(delta_guard + 20, tau_target - half_w)
        hi = min(len(x) // 3,     tau_target + half_w)
        candidates = [
            int(t) for t in np.linspace(lo, hi, n_side * 3)
            if abs(int(t) - tau_target) > delta_guard
        ][:n_side]

        if not candidates:
            return 1.0

        sidelobes = [single_caf(t) for t in candidates]
        median_sl = float(np.median(sidelobes))
        return peak / (median_sl + 1e-12)

    # =========================================================================
    # 公开接口
    # =========================================================================
    def run_spectral_audit(self, iq_data_buffer) -> tuple:
        """
        对输入 IQ 缓冲区执行完整的 CAF-FFT 循环频率审计。

        检测流程（四级漏斗）：
          Level 1  帧级 CAF 扫描 → 每帧提取 OcuSync 30k/15k 两通道 NCC 峰值
          Level 2  联合统计判决 → combined = α·peak + (1-α)·avg > 动态阈值
          Level 3  τ 域 PSR 验证 → 峰值旁瓣比确认 CP Delta 冲激特征
          Level 4  α 域 CFS 验证 → 循环频率集中度排除宽频扰动（可选加严）

        WiFi 干扰抑制原理（无需任何启发式规则）：
          IEEE 802.11 的循环频率为 250 kHz，在 OcuSync 的 18~32 kHz 扫描窗口
          之外，CAF 正交性使其泄漏幅度 ≈ 0.028%（即使 WiFi 强 20 dB 也无碍）。

        Parameters
        ----------
        iq_data_buffer : array-like
            AD9364 ADC 原始整数 IQ 数据（int16，复数或交织格式均可）

        Returns
        -------
        (bool, float) : (检测结果, 最大 NCC 系数)
        """
        buf           = np.asarray(iq_data_buffer)
        total_samples = len(buf)
        step_size     = int(self.CHUNK_SIZE * (1.0 - self.OVERLAP))

        # ── Level 1：逐帧 CAF-FFT 扫描 ─────────────────────────────────────
        frames_30k: list[tuple] = []   # (peak_ncc, best_alpha, cfs)
        frames_15k: list[tuple] = []
        chunks_by_frame: list   = []

        for i in range(0, total_samples - self.CHUNK_SIZE, step_size):
            chunk = buf[i : i + self.CHUNK_SIZE]
            x, pwr = self._prepare_chunk(chunk)
            if x is None:
                continue

            # OcuSync 30kHz 通道
            spec_30k = self._compute_caf_spectrum(x, self.TAU_OCUSYNC_30K, pwr)
            ncc_30k, alpha_30k, cfs_30k = self._extract_ncc_in_range(
                spec_30k, self.CHUNK_SIZE, self.ALPHA_SCAN_30K
            )

            # OcuSync 15kHz 通道
            spec_15k = self._compute_caf_spectrum(x, self.TAU_OCUSYNC_15K, pwr)
            ncc_15k, alpha_15k, cfs_15k = self._extract_ncc_in_range(
                spec_15k, self.CHUNK_SIZE, self.ALPHA_SCAN_15K
            )

            frames_30k.append((ncc_30k, alpha_30k, cfs_30k))
            frames_15k.append((ncc_15k, alpha_15k, cfs_15k))
            chunks_by_frame.append(x)

        if not frames_30k:
            print("  [S3-v3] WARNING: no valid frames (all below power gate)")
            return False, 0.0

        arr_30k = np.array([f[0] for f in frames_30k])
        arr_15k = np.array([f[0] for f in frames_15k])

        peak_30k = float(arr_30k.max())
        peak_15k = float(arr_15k.max())
        avg_30k  = float(arr_30k.mean())
        avg_15k  = float(arr_15k.mean())

        # 联合统计量：兼顾突发峰值（高 peak_weight）与持续弱信号（高 avg 权重）
        combined_30k = self.PEAK_WEIGHT * peak_30k + (1.0 - self.PEAK_WEIGHT) * avg_30k
        combined_15k = self.PEAK_WEIGHT * peak_15k + (1.0 - self.PEAK_WEIGHT) * avg_15k

        # WiFi 通道监控（仅日志，不参与判决）
        best_frame_x = chunks_by_frame[int(arr_30k.argmax())]
        _, pwr_best  = self._prepare_chunk(best_frame_x)   # 再次归一化，保持一致
        wifi_ncc = 0.0
        if pwr_best and pwr_best > self.MIN_POWER_GATE:
            spec_wifi = self._compute_caf_spectrum(best_frame_x, self.TAU_WIFI, pwr_best)
            k_wifi = max(1, int(np.round(self.ALPHA_WIFI_HZ / (self.sample_rate / len(spec_wifi)))))
            k_wifi = min(k_wifi, len(spec_wifi) - 1)
            wifi_ncc = float(spec_wifi[k_wifi])

        print(
            f"  [S3-v3] {len(frames_30k)} frames | "
            f"OcuSync30k: peak={peak_30k*100:.2f}% avg={avg_30k*100:.2f}% "
            f"→ combined={combined_30k*100:.2f}% (th={self.THRESHOLD_30K*100:.1f}%) | "
            f"OcuSync15k: peak={peak_15k*100:.2f}% avg={avg_15k*100:.2f}% "
            f"→ combined={combined_15k*100:.2f}% (th={self.THRESHOLD_15K*100:.1f}%) | "
            f"WiFi@250kHz(monitor)={wifi_ncc*100:.2f}%"
        )

        # ── Level 2：联合统计量一级门限 ────────────────────────────────────
        triggered_ch  = None    # '30k' 或 '15k'
        triggered_score = 0.0

        if combined_30k >= self.THRESHOLD_30K:
            if combined_30k >= combined_15k or combined_15k < self.THRESHOLD_15K:
                triggered_ch    = '30k'
                triggered_score = combined_30k
            else:
                triggered_ch    = '15k'
                triggered_score = combined_15k
        elif combined_15k >= self.THRESHOLD_15K:
            triggered_ch    = '15k'
            triggered_score = combined_15k

        if triggered_ch is None:
            print("  [S3-v3] Below threshold -- no UAV detected.")
            return False, max(combined_30k, combined_15k)

        # 取触发通道的最强帧和对应参数
        if triggered_ch == '30k':
            best_idx     = int(arr_30k.argmax())
            tau_t        = self.TAU_OCUSYNC_30K
            alpha_best   = frames_30k[best_idx][1]
            cfs_best     = frames_30k[best_idx][2]
            label        = "OcuSync 30kHz (Mini 4 Pro / Mavic 3)"
        else:
            best_idx     = int(arr_15k.argmax())
            tau_t        = self.TAU_OCUSYNC_15K
            alpha_best   = frames_15k[best_idx][1]
            cfs_best     = frames_15k[best_idx][2]
            label        = "OcuSync 15kHz (Mini 3 / Air 2S)"

        best_x = chunks_by_frame[best_idx]
        _, pwr_best = self._prepare_chunk(best_x)
        if pwr_best is None:
            pwr_best = self.MIN_POWER_GATE

        # ── Level 3：τ 域 PSR 验证 ──────────────────────────────────────────
        # 根据 WiFi 监控值动态调整 PSR 要求：
        # wifi_ncc > 0.01 说明 250kHz 通道有能量（WiFi 存在），但不代表 OcuSync 误报，
        # 仅作为上调 PSR 严格程度的依据。
        psr_th = self.PSR_THRESHOLD_WIFI if wifi_ncc > 0.010 else self.PSR_THRESHOLD
        psr    = self._compute_psr(best_x, pwr_best, tau_t, alpha_best)

        print(
            f"  [S3-v3] PSR check: tau={tau_t}, alpha={alpha_best/1e3:.1f}kHz, "
            f"PSR={psr:.2f}x (th={psr_th:.1f}x), CFS={cfs_best:.2f}x (th={self.CFS_THRESHOLD:.1f}x)"
        )

        if psr < psr_th:
            print(
                f"  [S3-v3] PSR check failed ({psr:.2f}x < {psr_th:.1f}x) -- "
                f"flat delay peak, classified as wideband/SMPS interference. Alert suppressed."
            )
            return False, triggered_score

        # ── Level 4：α 域 CFS 验证（循环频率集中度）────────────────────────
        # 真实 OcuSync：循环频率高度集中在特定 α 处（CFS >> 1）
        # 宽带干扰（如杂散谐波）：CAF 谱平坦（CFS ≈ 1）
        if cfs_best < self.CFS_THRESHOLD:
            print(
                f"  [S3-v3] CFS check failed ({cfs_best:.2f}x < {self.CFS_THRESHOLD:.1f}x) -- "
                f"diffuse cycle spectrum, not OcuSync. Alert suppressed."
            )
            return False, triggered_score

        print(f"  [S3-v3] CONFIRMED: {label}")
        print(f"           NCC={triggered_score*100:.2f}%  alpha={alpha_best/1e3:.2f}kHz  "
              f"PSR={psr:.1f}x  CFS={cfs_best:.1f}x")

        self._save_snapshot(best_x, pwr_best, tau_t, alpha_best,
                            triggered_score, triggered_ch, label)

        return True, triggered_score

    # =========================================================================
    # 诊断快照（保留 v2.x 功能，升级为 CAF 谱可视化）
    # =========================================================================
    def _save_snapshot(
        self,
        x: np.ndarray,
        power: float,
        tau: int,
        alpha_best_hz: float,
        score: float,
        channel: str,
        label: str,
    ):
        """Generate and save CAF spectrum snapshot (two subplots: 30k and 15k channels)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['font.family'] = ['DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            import matplotlib.pyplot as plt, os

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            for ax, tau_plot, alpha_range, ch_label in [
                (axes[0], self.TAU_OCUSYNC_30K, self.ALPHA_SCAN_30K, '30kHz'),
                (axes[1], self.TAU_OCUSYNC_15K, self.ALPHA_SCAN_15K, '15kHz'),
            ]:
                spec = self._compute_caf_spectrum(x, tau_plot, power)
                N    = len(spec)
                f_hz = np.arange(N) * (self.sample_rate / N)
                mask = f_hz < 350_000
                ax.semilogy(f_hz[mask] / 1e3, spec[mask] + 1e-6,
                            color='#FF5722', linewidth=1.0, label='CAF-NCC spectrum')
                ax.axvspan(alpha_range[0]/1e3, alpha_range[1]/1e3,
                           alpha=0.15, color='lime', label='OcuSync scan window')
                ax.axvline(self.ALPHA_WIFI_HZ / 1e3, color='orange',
                           linestyle='--', lw=1.2, label='WiFi 250kHz')
                ax.axvline(alpha_best_hz / 1e3 if channel == ('30k' if ch_label=='30kHz' else '15k')
                           else 0, color='cyan', linestyle=':', lw=1.5, label='alpha_best')
                ax.axhline(self.THRESHOLD_30K if ch_label=='30kHz' else self.THRESHOLD_15K,
                           color='red', linestyle='--', lw=1, label='Threshold')
                ax.set_title(f"CAF spectrum  tau={tau_plot}  ({ch_label} channel)", fontsize=11)
                ax.set_xlabel("Cycle frequency alpha (kHz)")
                ax.set_ylabel("NCC (log)")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
                ax.set_xlim(0, 350)

            fig.suptitle(
                f"S3 CAF-FFT Audit -- {label}\n"
                f"NCC={score*100:.2f}%  alpha={alpha_best_hz/1e3:.2f}kHz",
                fontsize=12, fontweight='bold'
            )
            plt.tight_layout()

            db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'database', 'alert_images')
            os.makedirs(db_dir, exist_ok=True)
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(db_dir, f'S3_CAF_{channel}_{ts}.png')
            plt.savefig(path, dpi=130)
            plt.close()
            print(f"  [S3-v3] Snapshot saved: {path}")
        except Exception as e:
            print(f"  [S3-v3] Snapshot failed (non-critical): {e}")

