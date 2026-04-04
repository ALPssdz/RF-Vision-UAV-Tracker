import adi
import numpy as np
import time

class RF_Stage2_Dwell:
    """
    Cognitive RF Tier 2: Dwell Phase and Vision Object Processing.
    对一阶网络提交的占优活跃带频分通道进行连续不断的驻留式阵列捕集。
    负责把长序时间轴多维物理采样列进行阵列截取缩编，生成适用于 YOLO 卷积核体系等效运算维度的 640x640 规整张量缓冲块。
    """
    def __init__(self, sdr_instance):
        self.sdr = sdr_instance
        self.buffer_size = 16384
        self.window = np.blackman(self.buffer_size)
        
        self.target_width = 640
        self.target_height = 640
        
        # --- 降维约束级收割切分法则参数化配置 ---
        # 16384 // 640 = 25，生成数组列末会保留 384 无效位点边界。
        # 取中将边界区域截短 192 特征点实施齐整对齐对等切分。
        self.pool_size = self.buffer_size // self.target_width
        self.trim_side = (self.buffer_size - (self.pool_size * self.target_width)) // 2
        
        # 控制像素阈值域色彩极值配置条件
        self.vmin = -70
        self.vmax = 30

    def _convert_to_1d_pooled_db(self, complex_iq):
        """
        激活基带快速傅里叶时频换算 (FFT) 并执行最大池化结构约束操作，将全带特征信息矢量块收缩合并进入 640 个有效空间像素靶点限度。
        """
        # ==============================================================
        # 硬件结构基底层级参数差异配准补偿机制:
        # 区别于标准通用 USRP 1.0 的最大单位域内浮点规范化表示输出特征，PlutoSDR 的发送下行 IQ 限制为定点高阶大数字边界 [-32768, 32767]。
        # 为符合 (-70, 30) dB 分界量程尺度，必需施加强制除法偏移约简 / 32768.0 的算术除以消除大量热图严重过白曝光突起特征的显现。
        # ==============================================================
        normalized_iq = complex_iq / 32768.0 
        
        windowed_data = normalized_iq * self.window
        fft_data = np.fft.fftshift(np.fft.fft(windowed_data))
        power_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        # >> 组件特征提取缩放及横向裁剪边缘规整匹配
        trimmed_db = power_db[self.trim_side : -self.trim_side]
        reshaped = trimmed_db.reshape((self.target_width, self.pool_size))
        
        # 应用最大池化操作算子（取代单纯平滑均值处理），保留绝对短瞬基带微弱窄跳突起信号不至于丢失。
        pooled_1d = np.max(reshaped, axis=1)
        
        return pooled_1d

    def generate_waterfall_tensor(self, center_freq):
        """
        以矩阵填充迭代产生出带有预标记属性的 Numpy 640x640 类型数据流列结构数组，
        并支持无接口阻碍传唤 NPU/YOLO 相关张量计算端点进行推理。
        """
        self.sdr.rx_lo = int(center_freq)
        _ = self.sdr.rx()
        _ = self.sdr.rx()
        
        waterfall = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        
        for idx in range(self.target_height):
            iq_data = self.sdr.rx()
            waterfall[idx, :] = self._convert_to_1d_pooled_db(iq_data)
            
        waterfall_clipped = np.clip(waterfall, self.vmin, self.vmax)
        waterfall_norm = ((waterfall_clipped - self.vmin) / (self.vmax - self.vmin) * 255.0)
        waterfall_uint8 = waterfall_norm.astype(np.uint8)
        
        # 运用 OpenCV 色彩算子映射单通道物理分布极值平面为符合通例的仿生类红外三通道矩阵
        import cv2
        waterfall_bgr = cv2.applyColorMap(waterfall_uint8, cv2.COLORMAP_HOT)
        
        return waterfall_bgr
