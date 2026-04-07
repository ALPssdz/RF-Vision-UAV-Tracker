import adi
import numpy as np
import time
import cv2


class RF_Stage2_Dwell:
    """
    射频检测第二级：IQ 驻留采集与频谱瀑布图生成模块。

    本模块对 AD9364 SDR 前端执行单次大块 DMA 采集，通过向量化短时傅里叶变换
    （Vectorized STFT）将原始 IQ 数据转换为 640×640 BGR 伪彩色频谱瀑布图张量，
    供后续 YOLOv8 推理端点消费。

    采用批量 DMA 采集策略（单次调用 rx() 获取全部样本）以规避逐行轮询模式下
    USB 传输速率不足导致的 DMA 溢出（表现为瀑布图横向条带伪影）。
    """

    def __init__(self, sdr_instance):
        """
        Parameters
        ----------
        sdr_instance : adi.ad9364
            已完成参数配置的 AD9364 SDR 实例，由上层 RFToolchain 注入。
        """
        self.sdr = sdr_instance
        self.target_width  = 640   # 输出图像宽度（像素），与 YOLOv8 输入尺寸一致
        self.target_height = 640   # 输出图像高度（像素），对应 STFT 帧数
        self.fft_size      = 2048  # 单帧 FFT 点数
        #   频率分辨率: 40 MSps / 2048 = 19.5 kHz/bin
        #   OcuSync 30kHz 子载波间距 ≈ 1.5 bin（足够区分，同时使 50% 重叠
        #   帧数量恰好覆盖 2,621,440 样本 = 2559 帧 > 640，可选 1/4 帧间隔）
        self.hop_size      = self.fft_size // 2   # 50% 重叠，hop = 1024

        # Blackman 窗：第一旁瓣 -58 dB，适合抑制 OFDM 宽带旁瓣
        self.window    = np.blackman(self.fft_size).astype(np.float32)

        # 功率映射范围（dBFS，未归一化的原始 FFT 幅度对数）
        # N=2048 Blackman vs N=4096 Blackman：噪声底前者低约 3 dB（sqrt(2)）
        # vmin 从 -60 小幅提升至 -63、vmax 从 +30 调至 +27 补偿该差异
        self.vmin = -63
        self.vmax =  27

    def generate_waterfall_tensor(self, center_freq: float) -> np.ndarray:
        """
        对指定中心频率执行一次完整的 IQ 驻留采集并生成频谱瀑布图张量。

        采集流程：
          1. 切换 SDR 本振至目标频率；
          2. 丢弃残留缓冲（消除切频过渡态干扰）；
          3. 单次 DMA 突发采集 2,621,440 个复数样本（约 65 ms 时窗）；
          4. 直流偏置校正：减去批次均值，消除本振直流泄漏；
          5. 向量化 STFT：将 IQ 序列重整为 (640, 4096) 矩阵，
             施加 Blackman 窗后批量执行 FFT；
          6. 频域降采样：最大值池化将 4096 频点压缩至 640 列；
          7. 伪彩色映射：归一化后应用 HOT 色盘生成 BGR 三通道图像。

        Parameters
        ----------
        center_freq : float
            本次驻留的接收中心频率（Hz）。

        Returns
        -------
        np.ndarray
            形状为 (640, 640, 3)、dtype 为 uint8 的 BGR 频谱瀑布图张量。
        """
        self.sdr.rx_lo = int(center_freq)

        # 丢弃切换本振后残留在 DMA 缓冲区中的旧数据帧
        try:
            _ = self.sdr.rx()
            _ = self.sdr.rx()
        except Exception:
            pass

        # 单次突发采集
        raw_iq = self.sdr.rx()
        self.last_buffer_iq = raw_iq  # 保存原始 IQ 供 S3 循环谱模块复用

        # 归一化 + 直流偏置校正
        # 除以 32768 使 ADC 满量程对应 ±1.0；减均值消除 DC 泄漏
        iq = raw_iq.astype(np.complex64) / 32768.0
        iq -= iq.mean()

        # ── 向量化 STFT（50% 重叠） ──────────────────────────────────────────
        # 使用 stride_tricks 构建重叠帧视图，避免数据复制开销
        # 帧数量 n_frames = (N - fft_size) // hop_size + 1
        #   = (2,621,440 - 2048) // 1024 + 1 ≈ 2559 帧
        # 从 2559 帧中等间隔取 640 帧（stride 约 4 帧）生成瀑布图
        N        = len(iq)
        n_frames_total = (N - self.fft_size) // self.hop_size + 1
        # 等间隔从所有帧中选出 target_height 帧
        indices  = np.linspace(0, n_frames_total - 1,
                               self.target_height, dtype=int)
        starts   = indices * self.hop_size

        # 构建帧矩阵：(target_height, fft_size)
        frames   = np.array([iq[s: s + self.fft_size] for s in starts],
                            dtype=np.complex64)

        # 施加 Blackman 窗并执行批量 FFT
        # 不除以 _win_gain：保持原始 FFT 幅度尺度（与 vmin/vmax 校准值匹配）
        # （除以 _win_gain 会将幅度压低 59 dB，导致大多数信号落在 vmin 之下，图像全黑）
        windowed  = frames * self.window
        fft_data  = np.fft.fftshift(
            np.fft.fft(windowed, axis=1), axes=1
        )    # 不除以 _win_gain

        # 幅度谱（dBFS，未归一化）
        power_db  = 20.0 * np.log10(np.abs(fft_data).astype(np.float32) + 1e-12)

        # ── 频域降采样：线性功率均值池化 ──────────────────────────────────────
        # 旧版使用 np.max()（最大值池化），其会将 Blackman 旁瓣（应 -58 dB）
        # 抬高至与主瓣同量级，导致 OFDM 信号视觉上向周围频率大幅扩散。
        #
        # 正确做法：在线性功率域取均值，保留真实功率谱形状：
        #   P_lin[i, k] = 10^(power_db[i,k] / 10)      (dBFS → 线性功率)
        #   P_avg[i, j] = mean(P_lin[i, j*M : (j+1)*M]) (M bin 均值)
        #   result[i, j] = 10*log10(P_avg[i, j])         (线性功率 → dBFS)
        #
        # 效果：旁瓣能量在均值后仍保持在真实水平，主瓣与旁瓣对比度恢复正常
        pool_size  = self.fft_size // self.target_width   # = 3
        trim_side  = (self.fft_size - pool_size * self.target_width) // 2
        trimmed_db = power_db[:, trim_side: self.fft_size - trim_side]

        # dBFS → 线性功率（注意：power_db = 20*log10(|X|)，所以 /10 对应功率）
        lin_power    = 10.0 ** (trimmed_db / 10.0)
        pool_shaped  = lin_power.reshape(
            self.target_height, self.target_width, pool_size
        )
        waterfall_lin = np.mean(pool_shaped, axis=2)           # 线性功率均值
        waterfall = 10.0 * np.log10(waterfall_lin + 1e-20)     # 回到 dBFS

        # ── 伪彩色映射 ─────────────────────────────────────────────────────
        waterfall_clipped = np.clip(waterfall, self.vmin, self.vmax)
        waterfall_norm    = (
            (waterfall_clipped - self.vmin)
            / (self.vmax - self.vmin)
            * 255.0
        )
        waterfall_uint8 = waterfall_norm.astype(np.uint8)
        waterfall_bgr   = cv2.applyColorMap(waterfall_uint8, cv2.COLORMAP_VIRIDIS)

        return waterfall_bgr
