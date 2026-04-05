"""
S3 误报根因诊断工具 (False Positive Root-Cause Analyzer)
=========================================================
在没有无人机的环境中，对真实采集的背景噪声执行全谱循环相关扫描。
用于精准识别哪些环境干扰源正在 Tau=1333 或 Tau=2667 处产生虚假峰值。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os

# =========================================================================
# 【配置区】
# =========================================================================
SDR_URI         = "ip:192.168.31.10"
SAMPLE_RATE     = int(40e6)
RX_GAIN         = 50           # dB
CAPTURE_SECONDS = 5            # 捕获秒数
BUFFER_SIZE     = 2621440      # 单次 DMA 拉取长度（65ms）

# DJI Mini 3 5.8GHz 模式工作频段约 5745~5825MHz
# 以 40MHz 带宽覆盖: 5745→[5725-5765], 5785→[5765-5805], 5825→[5805-5845]
CENTER_FREQS    = [5745e6, 5785e6, 5825e6]

# 目标 Tau 特征点
TAU_MINI4   = 1333
TAU_MINI3   = 2667
TAU_WIFI_CP = 128

# 全谱扫描范围（用于绘图）
TAU_SCAN_START  = 50
TAU_SCAN_END    = 3500
TAU_SCAN_STEP   = 5  # 步长（越小越精细，越慢）

# 诊断结果输出路径
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "alert_images")
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================================================

def compute_cp_score(iq, tau):
    """标准化时移自相关，与 S3 内核完全一致。"""
    dc_removed = iq - np.mean(iq)
    iq_main    = dc_removed[tau:]
    iq_delayed = dc_removed[:-tau]
    corr       = np.abs(np.mean(iq_main * np.conj(iq_delayed)))
    pwr        = np.mean(np.abs(iq_main)**2)
    return corr / (pwr + 1e-12), pwr   # 同时返回功率供诊断

def capture_background(uri, freq_hz, n_captures):
    """从 PlutoSDR 采集多段背景 IQ 数据。"""
    try:
        import adi
        sdr = adi.Pluto(uri)
        sdr.sample_rate = SAMPLE_RATE
        sdr.rx_rf_bandwidth = SAMPLE_RATE
        sdr.rx_hardwaregain_control_mode = 'manual'
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = BUFFER_SIZE
        sdr.rx_lo = int(freq_hz)

        # 冲刷历史残影
        for _ in range(3):
            _ = sdr.rx()
        time.sleep(0.1)

        buffers = []
        for i in range(n_captures):
            raw = sdr.rx()
            buffers.append(raw.astype(np.complex64) / 32768.0)
            print(f"  [{freq_hz/1e6:.0f}MHz] 已采集 {i+1}/{n_captures} 段...")
        return buffers

    except Exception as e:
        print(f"[!] SDR 连接失败: {e}")
        print("[!] 将切换到离线模式（使用一段纯随机高斯白噪声仿真背景）...")
        # 离线模式：使用纯白噪声测试
        sigma = 0.001  # 很低的底噪功率
        buffers = [
            (sigma * (np.random.randn(BUFFER_SIZE) + 1j * np.random.randn(BUFFER_SIZE))).astype(np.complex64)
            for _ in range(n_captures)
        ]
        return buffers

def run_diagnosis():
    n_captures = int(CAPTURE_SECONDS * SAMPLE_RATE / BUFFER_SIZE) + 1

    all_results = {}

    for freq in CENTER_FREQS:
        print(f"\n[DIAG] === 正在捕获扇区 {freq/1e6:.0f}MHz 的无人机 OFF 背景 ===")
        buffers = capture_background(SDR_URI, freq, n_captures)

        # ---- 全谱循环相关扫描 ----
        tau_list   = np.arange(TAU_SCAN_START, TAU_SCAN_END, TAU_SCAN_STEP)
        score_avg  = np.zeros(len(tau_list))
        score_max  = np.zeros(len(tau_list))
        pwr_list   = []

        print(f"  [SCAN] 正在对 {len(buffers)} 段数据进行全谱暴力扫描 (Tau={TAU_SCAN_START}~{TAU_SCAN_END})...")
        for buf in buffers:
            # 截取中间最精华的一个切片，避免冷启动噪声
            chunk_size = min(200000, len(buf))
            chunk = buf[BUFFER_SIZE//2 : BUFFER_SIZE//2 + chunk_size]

            for i, tau in enumerate(tau_list):
                s, p = compute_cp_score(chunk, int(tau))
                score_avg[i] += s
                score_max[i]  = max(score_max[i], s)
            pwr_list.append(np.mean(np.abs(chunk)**2))

        score_avg /= len(buffers)
        mean_power = np.mean(pwr_list)

        all_results[freq] = {
            "tau": tau_list,
            "score_avg": score_avg,
            "score_max": score_max,
            "mean_power": mean_power,
        }

        # 打印关键 Tau 点的真实分数
        for tau_target, name in [(TAU_MINI4, "Mini4/OcuSync30k"), (TAU_MINI3, "Mini3/OcuSync15k"), (TAU_WIFI_CP, "Wi-Fi CP")]:
            idx = np.argmin(np.abs(tau_list - tau_target))
            print(f"  [{freq/1e6:.0f}MHz] Tau={tau_target:4d} ({name}): "
                  f"平均分={score_avg[idx]*100:.2f}%  最大分={score_max[idx]*100:.2f}%")

        print(f"  [{freq/1e6:.0f}MHz] 背景信号平均功率: {mean_power:.6f}  ({10*np.log10(mean_power+1e-12):.1f} dBFS)")

    # ---- 绘图 ----
    n_freqs = len(all_results)
    fig = plt.figure(figsize=(16, 5 * n_freqs))
    gs  = gridspec.GridSpec(n_freqs, 1, figure=fig)

    colors = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0']  # 支持 4 个扇区

    for idx, (freq, result) in enumerate(all_results.items()):
        ax = fig.add_subplot(gs[idx])
        tau   = result["tau"]
        s_avg = result["score_avg"]
        s_max = result["score_max"]
        pwr   = result["mean_power"]

        ax.fill_between(tau, s_avg, alpha=0.3, color=colors[idx])
        ax.plot(tau, s_avg, color=colors[idx], linewidth=1.5, label='平均分 (穿越率最重要)')
        ax.plot(tau, s_max, color=colors[idx], linewidth=0.8, linestyle='--', alpha=0.6, label='最大分 (极端情况)')

        # 标记目标 Tau
        for tau_t, name, color in [
            (TAU_MINI4, "1333 (Mini4)", 'blue'),
            (TAU_MINI3, "2667 (Mini3)", 'green'),
            (TAU_WIFI_CP, "128 (Wi-Fi)", 'orange'),
        ]:
            ax.axvline(tau_t, color=color, linestyle=':', alpha=0.8, linewidth=1.5)
            ax.text(tau_t+10, ax.get_ylim()[1]*0.85, name, color=color, fontsize=8, rotation=90)

        # 标记当前阈值
        ax.axhline(0.045, color='red', linestyle='--', linewidth=1.5, label='当前告警阈值 (4.5%)')
        ax.axhline(0.045 * 0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='候选安全线 (2.25%)')

        ax.set_title(
            f"背景循环谱 (无无人机) @ {freq/1e6:.0f}MHz  |  "
            f"背景功率={pwr:.5f} ({10*np.log10(pwr+1e-12):.1f} dBFS)",
            fontsize=12
        )
        ax.set_xlabel("延迟 Tau (样本数)")
        ax.set_ylabel("CP 相关分数")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(TAU_SCAN_START, TAU_SCAN_END)
        ax.set_ylim(0, min(0.20, max(s_max) * 1.3 + 0.01))

    plt.suptitle("S3 FALSE POSITIVE ROOT-CAUSE ANALYSIS\n(UAV OFF, Environment Background Only)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "s3_false_positive_diagnosis.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"\n[DIAG] 完整诊断图谱已输出: {out_path}")

    # ---- 给出具体修复建议 ----
    print("\n[DIAG] ===== 自动诊断意见 =====")
    for freq, result in all_results.items():
        tau   = result["tau"]
        s_avg = result["score_avg"]
        s_max = result["score_max"]
        pwr   = result["mean_power"]

        def get_score(tau_t):
            idx = np.argmin(np.abs(tau - tau_t))
            return s_avg[idx], s_max[idx]

        s1333_avg, s1333_max = get_score(TAU_MINI4)
        s2667_avg, s2667_max = get_score(TAU_MINI3)

        print(f"\n  [{freq/1e6:.0f}MHz] 背景环境分析:")
        if pwr < 1e-5:
            print(f"  ⚠️  归一化陷阱警告：背景功率极低 ({pwr:.2e})，噪声涨落会被功率归一放大！")
            safe_threshold = max(s_max) * 1.5
            print(f"     → 建议统一要求 power_main > 1e-4 才计算，否则直接返回 0.0")
        if s1333_max > 0.04:
            print(f"  🔴 Tau=1333 (30kHz) 背景最高分 {s1333_max*100:.2f}%，超过安全线！")
            print(f"     可能原因: 5G NR 基站 / SMPS 30kHz 开关电源纹波")
            print(f"     建议阈值: > {s1333_max*1.5*100:.1f}% 才告警")
        if s2667_max > 0.04:
            print(f"  🔴 Tau=2667 (15kHz) 背景最高分 {s2667_max*100:.2f}%，超过安全线！")
            print(f"     可能原因: 4G LTE 基站溢出干扰（15kHz 子载波与 OcuSync 2.0 完全重合！）")
            print(f"     建议阈值: > {s2667_max*1.5*100:.1f}% 才告警")

if __name__ == "__main__":
    run_diagnosis()
