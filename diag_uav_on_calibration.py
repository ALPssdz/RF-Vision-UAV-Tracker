"""
S3 真机信号强度校准工具 (UAV-ON Signal Calibration)
====================================================
在无人机开机并与遥控器连接后运行。
测量真实 OcuSync 信号在 Tau=1333 和 Tau=2667 处的实际得分。
对比 diag_s3_false_positive.py 的背景得分，确定最终的告警阈值。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import time, os

SDR_URI      = "ip:192.168.31.10"
SAMPLE_RATE  = int(40e6)
RX_GAIN      = 50
BUFFER_SIZE  = 2621440
# DJI Mini 3 5.8GHz 三扇区全覆盖
CENTER_FREQS = [5745e6, 5785e6, 5825e6]

TAU_MINI4   = 1333
TAU_MINI3   = 2667

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "alert_images")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_cp_score(iq, tau, min_power=1e-5):
    dc = iq - np.mean(iq)
    m, d = dc[tau:], dc[:-tau]
    pwr = np.mean(np.abs(m)**2)
    if pwr < min_power:
        return 0.0, pwr
    corr = np.abs(np.mean(m * np.conj(d)))
    return corr / (pwr + 1e-12), pwr


def capture_uav_on(uri, freq_hz, n_captures=8):
    try:
        import adi
        sdr = adi.Pluto(uri)
        sdr.sample_rate = SAMPLE_RATE
        sdr.rx_rf_bandwidth = SAMPLE_RATE
        sdr.rx_hardwaregain_control_mode = 'manual'
        sdr.rx_hardwaregain_chan0 = RX_GAIN
        sdr.rx_buffer_size = BUFFER_SIZE
        sdr.rx_lo = int(freq_hz)
        for _ in range(3): _ = sdr.rx()
        time.sleep(0.1)
        bufs = []
        for i in range(n_captures):
            bufs.append(sdr.rx().astype(np.complex64) / 32768.0)
            print(f"  [{freq_hz/1e6:.0f}MHz] 采集 {i+1}/{n_captures}...")
        return bufs
    except Exception as e:
        print(f"[!] SDR 连接失败: {e}")
        return None


def run_calibration():
    print("=" * 60)
    print(" S3 真机信号校准工具 - 请确认：")
    print("   [1] 无人机已经开机")
    print("   [2] 遥控器已经开机")
    print("   [3] 两者已连接对频（绿灯常亮）")
    print("   [4] 无人机距离接收天线约 1~3 米")
    print("=" * 60)

    fig, axes = plt.subplots(len(CENTER_FREQS), 1, figsize=(14, 5 * len(CENTER_FREQS)))
    if len(CENTER_FREQS) == 1:
        axes = [axes]

    # 背景基线（第二次稳定测量 @ 5.8GHz 无人机 OFF）
    # 5745MHz: Tau=1333→3.18%, Tau=2667→2.63%
    # 5785MHz: Tau=1333→6.08%, Tau=2667→3.05%  ← 最坏情况
    # 5825MHz: 功率极低(7.4e-06)，MIN_POWER_GATE 自动拦截，忽略其读数
    known_bg_max = {5745e6: (0.0329, 0.0241), 5785e6: (0.0192, 0.0186), 5825e6: (0.0322, 0.0298)}

    for ax, freq in zip(axes, CENTER_FREQS):
        print(f"\n[CALIB] === 扇区 {freq/1e6:.0f}MHz ===")
        bufs = capture_uav_on(SDR_URI, freq)
        if bufs is None:
            ax.text(0.5, 0.5, f"SDR离线 @ {freq/1e6:.0f}MHz", ha='center', transform=ax.transAxes)
            continue

        tau_list = np.arange(50, 3500, 5)
        scores_all = np.zeros((len(bufs), len(tau_list)))

        print(f"  [SCAN] 全谱扫描中...")
        for bi, buf in enumerate(bufs):
            chunk = buf[BUFFER_SIZE//2 : BUFFER_SIZE//2 + 200000]
            for ti, tau in enumerate(tau_list):
                s, _ = compute_cp_score(chunk, int(tau))
                scores_all[bi, ti] = s

        score_avg = np.mean(scores_all, axis=0)
        score_max = np.max(scores_all, axis=0)

        # 关键 Tau 点
        def get_score(tau_t):
            idx = np.argmin(np.abs(tau_list - tau_t))
            return score_avg[idx], score_max[idx]

        s1333_avg, s1333_max = get_score(TAU_MINI4)
        s2667_avg, s2667_max = get_score(TAU_MINI3)
        bg_1333, bg_2667 = known_bg_max.get(freq, (0.14, 0.09))

        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  Tau=1333 (Mini4)  信号均值: {s1333_avg*100:5.2f}%  最大: {s1333_max*100:5.2f}%  ║")
        print(f"  ║  Tau=2667 (Mini3)  信号均值: {s2667_avg*100:5.2f}%  最大: {s2667_max*100:5.2f}%  ║")
        print(f"  ║  ─────────────────────────────────────  ║")
        print(f"  ║  背景(OFF): Tau=1333:{bg_1333*100:.2f}%  Tau=2667:{bg_2667*100:.2f}%  ║")

        # 推荐阈值：背景峰值 + 25% 信号-背景差的缓冲
        rec_th_1333 = bg_1333 + (s1333_max - bg_1333) * 0.5 if s1333_max > bg_1333 else bg_1333 * 1.8
        rec_th_2667 = bg_2667 + (s2667_max - bg_2667) * 0.5 if s2667_max > bg_2667 else bg_2667 * 1.8
        print(f"  ║  ─────────────────────────────────────  ║")
        print(f"  ║  ⭐ 推荐阈值: Tau=1333 → {rec_th_1333*100:.1f}%             ║")
        print(f"  ║  ⭐ 推荐阈值: Tau=2667 → {rec_th_2667*100:.1f}%             ║")
        print(f"  ╚══════════════════════════════════════════╝")

        # 绘图
        ax.fill_between(tau_list, score_avg, alpha=0.25, color='#FF5722')
        ax.plot(tau_list, score_avg, color='#FF5722', linewidth=1.5, label='UAV-ON 平均分')
        ax.plot(tau_list, score_max, color='#FF5722', linewidth=0.8, linestyle='--', alpha=0.6, label='UAV-ON 最大分')

        ax.axvline(TAU_MINI4, color='blue',  linestyle=':', linewidth=2)
        ax.axvline(TAU_MINI3, color='green', linestyle=':', linewidth=2)
        ax.axhline(bg_1333, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'背景峰值1333:{bg_1333*100:.1f}%')
        ax.axhline(bg_2667, color='gray', linestyle=':',  linewidth=1, alpha=0.7, label=f'背景峰值2667:{bg_2667*100:.1f}%')
        ax.axhline(rec_th_1333, color='red', linewidth=1.5, linestyle='-', label=f'推荐阈值1333:{rec_th_1333*100:.1f}%')

        ax.set_title(f"UAV-ON 循环谱 @ {freq/1e6:.0f}MHz — 信号与背景对比", fontsize=12)
        ax.set_xlabel("延迟 Tau (样本数)")
        ax.set_ylabel("CP 相关分数")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(50, 3500)

    plt.suptitle("S3 UAV-ON CALIBRATION\n(DJI Mini 3 Real Signal vs Environment Background)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "s3_uav_on_calibration.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"\n[CALIB] 校准图谱已保存: {out_path}")


if __name__ == "__main__":
    run_calibration()
