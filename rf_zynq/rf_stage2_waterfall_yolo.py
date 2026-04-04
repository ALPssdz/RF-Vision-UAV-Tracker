import adi
import numpy as np
import time

class RF_Stage2_Dwell:
    """
    认知射频第二阶段：驻留与凝视 (Dwell & Vision)
    在 Stage1 确定的高激活信道上，高速抓取连续帧。
    完成1维频率池化降维，组合为适用于 YOLO (640x640) 的图像张量内存块。
    """
    def __init__(self, sdr_instance):
        # 接收外部传入的 SDR 对象，防止重开占用
        self.sdr = sdr_instance
        self.buffer_size = 16384
        self.window = np.blackman(self.buffer_size)
        
        # YOLO 图片尺寸目标
        self.target_width = 640
        self.target_height = 640
        
        # --- 降维核心参数公式推导 ---
        # 16384 // 640 = 25，多出 384 个点。
        # 为保留最核心频谱，我们从两侧各砍掉 192 个点。
        self.pool_size = self.buffer_size // self.target_width
        self.trim_side = (self.buffer_size - (self.pool_size * self.target_width)) // 2
        
        # 色度灰度化映射限度，防止极大的尖峰让底噪全部沉沦为黑色影响 YOLO 判断
        # 可视化参数与前 sdr_burst2 兼容
        self.vmin = -70
        self.vmax = 30

    def _convert_to_1d_pooled_db(self, complex_iq):
        """
        进行 FFT 运算并将 16384 个频点通过最大池化坍缩到 640 个宽度的物理块
        """
        # ==============================================================
        # 🚨 极其致命的底层物理硬件差异补丁！
        # 训练基底用的 USRP 数据是最高为 1.0 的归一化浮点数，最高功率仅 30dB。
        # 而实时抓取的 PlutoSDR 返回的是 [-32768, +32767] 的超大整型！
        # 若不在此处除以 32768.0 归一化，其算出的计算功率将高达 +130dB 以上！
        # 导致所有数值强行破顶 vmax(30)，将热红外图全部烧成了死白！
        # ==============================================================
        normalized_iq = complex_iq / 32768.0 
        
        windowed_data = normalized_iq * self.window
        fft_data = np.fft.fftshift(np.fft.fft(windowed_data))
        power_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        # >> 核心裁剪与降维合并
        trimmed_db = power_db[self.trim_side : -self.trim_side]
        # 重塑为 (640, 25) 的矩阵
        reshaped = trimmed_db.reshape((self.target_width, self.pool_size))
        # 沿着每一段的频段宽度 (25个 FFT Bins) 获取最大峰值，彻底防止细小频段被均值淹没
        pooled_1d = np.max(reshaped, axis=1)
        
        return pooled_1d

    def generate_waterfall_tensor(self, center_freq):
        """
        参数化抓取并零拷贝生成 Numpy 640x640 张量。
        在 RK3588 上，直接将该矩阵(float32或uint8)过给 RKNN 执行推断即可。
        """
        self.sdr.rx_lo = int(center_freq)
        # 弃掉前两个脏 Buffer
        _ = self.sdr.rx()
        _ = self.sdr.rx()
        
        print(f"[Dwell] 锁定 {center_freq/1e6: .1f} MHz, 正在高频构建 640x640 内存张量...")
        
        waterfall = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        
        # 凝视生成图幅的时间理论值: 640 * 16384 / 40e6 = 0.262 s
        # 加入内存池化运算后整个块的时长应在 0.5s 上下 
        start = time.time()
        for idx in range(self.target_height):
            iq_data = self.sdr.rx()
            waterfall[idx, :] = self._convert_to_1d_pooled_db(iq_data)
            
        cost = time.time() - start
        
        # 归一化为 0-255 灰度单通道张量 (模拟图像，但不写盘)
        # 灰度计算公式：(DB - vmin) / (vmax - vmin) * 255
        waterfall_clipped = np.clip(waterfall, self.vmin, self.vmax)
        waterfall_norm = ((waterfall_clipped - self.vmin) / (self.vmax - self.vmin) * 255.0)
        waterfall_uint8 = waterfall_norm.astype(np.uint8)
        
        # ★ 极其关键的一步：将 numpy 的黑白张量用底层 C++ API 洗成 HOT 配色的 3 通道 BGR 张量
        import cv2
        waterfall_bgr = cv2.applyColorMap(waterfall_uint8, cv2.COLORMAP_HOT)
        
        print(f"✅ 张量生成就绪，凝视耗时: {cost:.3f} s，尺寸: {waterfall_bgr.shape}")
        
        # 直接将 waterfall_bgr 转给 YOLO 或者 RKNN API 执行推断
        return waterfall_bgr
        
if __name__ == "__main__":
    from rf_stage1_sweeper import RF_Stage1_Sweeper
    # 模拟串联测试
    sweeper = RF_Stage1_Sweeper()
    sweeper.initialize_sdr()
    target_f = sweeper.run_sweep_cycle()
    
    dweller = RF_Stage2_Dwell(sweeper.sdr)
    tensor = dweller.generate_waterfall_tensor(target_f)
    print("内存数据块头 10 个像素映射强度: ", tensor[0, :10])
