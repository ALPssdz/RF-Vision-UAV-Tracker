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
        self.target_height = 640   # 输出图像高度（像素），对应 STFT 行数
        self.fft_size      = 4096  # 单行 FFT 点数，决定频率分辨率
        self.window        = np.blackman(self.fft_size)  # Blackman 加权窗，抑制旁瓣

        # 功率映射范围（dBFS），与 YOLOv8 训练数据集的色彩空间保持一致
        self.vmin = -60   # 下限（噪声底）
        self.vmax =  30   # 上限（强信号峰值）

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

        # 丢弃切换本振后残留在 DMA 缓冲区中的旧数据帧（通常 1~2 个缓冲周期）
        try:
            _ = self.sdr.rx()
            _ = self.sdr.rx()
        except Exception:
            pass

        # 单次突发采集：一次调用获取全部 2,621,440 个样本，避免分批采集引入的相位不连续
        raw_iq = self.sdr.rx()
        self.last_buffer_iq = raw_iq  # 保存原始 IQ 供 S3 循环谱模块复用

        # 直流偏置校正：减去批次均值，消除 AD9364 本振直流泄漏
        normalized_iq = raw_iq / 32768.0
        normalized_iq = normalized_iq - np.mean(normalized_iq)

        # 裁剪或补零，使总长度对齐至 640 × 4096
        valid_length = self.target_height * self.fft_size
        if len(normalized_iq) > valid_length:
            normalized_iq = normalized_iq[:valid_length]
        elif len(normalized_iq) < valid_length:
            normalized_iq = np.pad(normalized_iq, (0, valid_length - len(normalized_iq)))

        # 向量化 STFT：将一维 IQ 序列重整为 (640, 4096) 矩阵，按行批量 FFT
        reshaped = normalized_iq.reshape((self.target_height, self.fft_size))
        windowed = reshaped * self.window
        fft_data  = np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)
        power_db  = 20 * np.log10(np.abs(fft_data) + 1e-12)

        # 频域降采样：最大值池化，将 4096 频点压缩至输出宽度 640
        pool_size  = self.fft_size // self.target_width
        trim_side  = (self.fft_size - pool_size * self.target_width) // 2
        trimmed_db = power_db[:, trim_side : -trim_side]
        pool_reshaped = trimmed_db.reshape((self.target_height, self.target_width, pool_size))
        waterfall = np.max(pool_reshaped, axis=2)

        # 归一化至 [0, 255] 并应用 HOT 伪彩色映射，生成可直接送入 YOLOv8 的 BGR 图像
        waterfall_clipped = np.clip(waterfall, self.vmin, self.vmax)
        waterfall_norm    = ((waterfall_clipped - self.vmin) / (self.vmax - self.vmin) * 255.0)
        waterfall_uint8   = waterfall_norm.astype(np.uint8)
        waterfall_bgr     = cv2.applyColorMap(waterfall_uint8, cv2.COLORMAP_HOT)

        return waterfall_bgr
