import adi
import numpy as np
import time

def generate_drone_mock_signal(fs, rs, num_symbols, fc_offset):
    """
    生成一个具有明显 alpha=500kHz 循环谱特征的基带无人机伪造信号。
    """
    sps = int(fs / rs) # 每个符号的采样点数 (Samples Per Symbol)
    
    # 随机生成 BPSK 符号阵列 [+1, -1]
    symbols = np.random.choice([-1.0, 1.0], size=num_symbols)
    
    # 使用矩形脉冲成型（直接重复填充），这会在频域引入主瓣和旁瓣，
    # 并在时不变的自相关域产生天然的 1/Ts 等距峰值（即我们的 500kHz 循环特征）
    baseband = np.repeat(symbols, sps)
    
    # 构建时间轴
    t = np.arange(len(baseband)) / fs
    
    # 为信源加载一个复数载波频偏，将其偏离中心零点，
    # 避免受到直交流 (DC) 泄露分量的能量干扰
    carrier = np.exp(1j * 2.0 * np.pi * fc_offset * t)
    signal_complex = baseband * carrier
    
    # PlutoSDR 发射端数据为有整型区间要求，故引入 20000.0 的数理放大倍距
    # 最大不可超过 32767
    signal_tx = signal_complex * 20000.0
    return signal_tx

def main():
    print("="*50)
    print("RF-Vision 靶机模拟器启动: PlutoSDR TX 映射")
    print("="*50)
    
    uri = "ip:192.168.31.20"
    print(f"[1] 正在通过网络端口握手独立靶机源硬件节点 {uri} ...")
    try:
        sdr = adi.Pluto(uri)
    except Exception as e:
        print(f"[错误] 无法找到网络中的靶机节点，请检查网线或 IP 地址映射: {e}")
        return
        
    # --- 发射机射频通道物理量标定 ---
    fs = int(40e6)
    sdr.sample_rate = fs
    sdr.tx_rf_bandwidth = fs
    
    # 设置载波发射点。我们主动向第一阶扫频段的中央 A 区 (2420 MHz) 发送靶场信号
    sdr.tx_lo = int(2420e6)
    sdr.tx_hardwaregain_chan0 = -10 # -10dB 具备极高信噪比表现量域
    
    # ★ 启动循环发包寄存队列，由硬件 FPGA 内部接管直接发送操作，减轻 USB 吞吐总线压迫
    sdr.tx_cyclic_buffer = True

    print("[2] 硬件端底座射频链路确权，正在演算宽频空间域发送张量...")
    
    # 依据第三方协议构建基础约束量: 500 ksps
    rs = 500e3
    num_symbols = 4000
    fc_offset = 2e6 # 频移至 +2MHz 带宽处，使其落在 2422 MHz 绝对空中链路段
    
    tx_data = generate_drone_mock_signal(fs, rs, num_symbols, fc_offset)
    
    print("[3] 开始在微波暗室/公开空中链路持续施加高能物理射频脉冲。")
    print(f"    传输信道: {sdr.tx_lo / 1e6} MHz (中心)")
    print(f"    特征偏移: +{fc_offset / 1e6} MHz")
    print(f"    验证锚点: Cyclostationary Feature Alpha = {rs / 1e3} kHz")
    print("\n>>> 已启用连续波映射，您可以随时打断当前控制台进程 (Ctrl+C) 以阻断发包。")
    
    # 执行 FPGA 发射压入
    sdr.tx(tx_data)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] 接收到中断命令，正在安全关停并注销信源设备资源配置...")
        # 解除循环阻断态发射
        sdr.tx_cyclic_buffer = False
        sdr.tx(np.zeros(1024)) # 推送死寂空间数据排空最后能量阵列
        try:
            sdr.tx_destroy_buffer()
        except:
            pass
        print("操作已规范化完成。")

if __name__ == "__main__":
    main()
