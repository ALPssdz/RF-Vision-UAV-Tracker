# -*- coding: utf-8 -*-
"""
calibrate_s3.py -- S3 CAF-FFT Auto Background Calibration
==========================================================
Automatically measures ambient NCC noise floor across all sectors,
derives optimal detection thresholds, and patches the source file.
No user interaction required -- runs fully autonomously.

Phases:
  Phase 1 -- Background noise baseline (UAV must be OFF)
             Capture IQ across all sectors, compute CAF-NCC floor.
  Phase 2 -- Threshold derivation & auto-patch
             th = max(HARD_FLOOR, bg_max x NOISE_MARGIN)
             Auto-patches rf_zynq/rf_stage3_cyclostationary.py
"""

import sys
import os
import re
import time
import numpy as np

# -- Project root (this file lives in rf_zynq/, root is two levels up) --------
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

S3_SOURCE = os.path.join(_PROJ_ROOT, "rf_zynq", "rf_stage3_cyclostationary.py")
OUT_DIR   = os.path.join(_PROJ_ROOT, "database", "alert_images")
os.makedirs(OUT_DIR, exist_ok=True)

# -- SDR parameters (aligned with backend_rk3588/config.py) ------------------
SDR_URI     = "ip:192.168.31.10"
SAMPLE_RATE = int(40e6)
RX_GAIN     = 50
BUFFER_SIZE = 2_621_440   # 65 ms @ 40 MSps
SECTORS_HZ  = [5745e6, 5785e6, 5825e6]
N_CAPTURES  = 6           # IQ captures per sector

# -- CAF scan parameters (identical to RF_Stage3_CycloAudit) ------------------
CHUNK_SIZE       = 200_000
TAU_30K, TAU_15K = 1333, 2667
ALPHA_SCAN_30K   = (18_000.0, 32_000.0)
ALPHA_SCAN_15K   = ( 9_000.0, 16_000.0)
MIN_POWER_GATE   = 1e-5

# -- Threshold derivation parameters ------------------------------------------
NOISE_MARGIN   = 5.0    # threshold = bg_max x NOISE_MARGIN
HARD_FLOOR_30K = 0.028  # 2.8% hard floor (13x theoretical noise floor)
HARD_FLOOR_15K = 0.022  # 2.2%

# =============================================================================
# Core CAF-FFT metric (identical algorithm to RF_Stage3_CycloAudit)
# =============================================================================
def _caf_ncc_peak(chunk_raw, tau, alpha_range):
    """
    Single-frame CAF-FFT normalized NCC peak.

    R_x^alpha(tau) via FFT of lag product z[n] = x[n] * conj(x[n-tau]).
    NCC[alpha] = |Z[k]| / (N_z * P_x)

    Returns (peak_ncc, best_alpha_hz)
    """
    x = chunk_raw.astype(np.complex64) / 32768.0
    x -= x.mean()
    power = float(np.mean(np.abs(x) ** 2))
    if power < MIN_POWER_GATE:
        return 0.0, alpha_range[0]

    z   = x[tau:] * np.conj(x[:-tau])
    N_z = len(z)
    Z   = np.fft.fft(z)
    ncc = np.abs(Z) / (N_z * (power + 1e-12))

    f_res = SAMPLE_RATE / N_z
    k_lo  = max(1,      int(np.round(alpha_range[0] / f_res)))
    k_hi  = min(N_z//2, int(np.round(alpha_range[1] / f_res)) + 1)

    if k_lo >= k_hi:
        return 0.0, alpha_range[0]

    seg      = ncc[k_lo:k_hi]
    best_idx = int(np.argmax(seg))
    return float(seg[best_idx]), float((k_lo + best_idx) * f_res)


# =============================================================================
# SDR capture
# =============================================================================
def _init_sdr(freq_hz):
    """Initialize SDR at given frequency. Returns sdr object or None."""
    try:
        import adi
        sdr = adi.Pluto(SDR_URI)
        sdr.sample_rate                  = SAMPLE_RATE
        sdr.rx_rf_bandwidth              = SAMPLE_RATE
        sdr.rx_hardwaregain_control_mode = 'manual'
        sdr.rx_hardwaregain_chan0        = RX_GAIN
        sdr.rx_buffer_size               = BUFFER_SIZE
        sdr.rx_lo                        = int(freq_hz)
        for _ in range(3):
            sdr.rx()   # flush stale buffers
        time.sleep(0.1)
        return sdr
    except Exception as e:
        print(f"  [!] SDR init failed: {e}")
        return None


def _capture_buffers(freq_hz, n):
    """Capture n IQ buffers. Returns empty list on failure."""
    sdr = _init_sdr(freq_hz)
    if sdr is None:
        return []
    bufs = []
    for i in range(n):
        bufs.append(sdr.rx())
        print(f"    [{freq_hz/1e6:.0f}MHz] captured {i+1}/{n}")
    return bufs


# =============================================================================
# Phase 1: Background noise baseline
# =============================================================================
def phase1_background():
    """
    Measure CAF-NCC ambient floor across all sectors (UAV must be OFF).

    Returns
    -------
    dict : {freq_hz: {'ncc_30k_max': float, 'ncc_15k_max': float,
                      'ncc_30k_avg': float, 'ncc_15k_avg': float}}
    """
    print("\n" + "=" * 60)
    print("  Phase 1 -- Background noise baseline (UAV OFF)")
    print("=" * 60)

    results = {}
    for freq in SECTORS_HZ:
        print(f"\n  [Sector {freq/1e6:.0f}MHz]")
        bufs = _capture_buffers(freq, N_CAPTURES)

        if not bufs:
            print(f"  SDR offline -- sector {freq/1e6:.0f}MHz skipped (using defaults)")
            results[freq] = {'ncc_30k_max': 0.030, 'ncc_15k_max': 0.025,
                             'ncc_30k_avg': 0.015, 'ncc_15k_avg': 0.013}
            continue

        ncc30_list, ncc15_list = [], []
        for buf in bufs:
            chunk = buf[BUFFER_SIZE // 2: BUFFER_SIZE // 2 + CHUNK_SIZE]
            n30, _ = _caf_ncc_peak(chunk, TAU_30K, ALPHA_SCAN_30K)
            n15, _ = _caf_ncc_peak(chunk, TAU_15K, ALPHA_SCAN_15K)
            ncc30_list.append(n30)
            ncc15_list.append(n15)

        r = {
            'ncc_30k_max': float(np.max(ncc30_list)),
            'ncc_15k_max': float(np.max(ncc15_list)),
            'ncc_30k_avg': float(np.mean(ncc30_list)),
            'ncc_15k_avg': float(np.mean(ncc15_list)),
        }
        results[freq] = r

        print(f"    OcuSync 30kHz: avg={r['ncc_30k_avg']*100:.2f}%  "
              f"max={r['ncc_30k_max']*100:.2f}%")
        print(f"    OcuSync 15kHz: avg={r['ncc_15k_avg']*100:.2f}%  "
              f"max={r['ncc_15k_max']*100:.2f}%")

    return results


# =============================================================================
# Phase 2: Threshold derivation & auto-patch
# =============================================================================
def _derive_thresholds(bg_results):
    """
    Derive optimal thresholds.

    Formula: th = max(HARD_FLOOR, bg_max x NOISE_MARGIN)
      - NOISE_MARGIN = 5.0  (5x safety margin above measured floor)
      - HARD_FLOOR   = absolute minimum (13x theoretical noise floor 1/sqrt(N))
      - Takes worst-case across all sectors

    Returns
    -------
    (th_30k, th_15k)
    """
    th30, th15 = [], []
    for freq in SECTORS_HZ:
        bg   = bg_results.get(freq, {})
        bg30 = bg.get('ncc_30k_max', HARD_FLOOR_30K / NOISE_MARGIN)
        bg15 = bg.get('ncc_15k_max', HARD_FLOOR_15K / NOISE_MARGIN)
        th30.append(max(HARD_FLOOR_30K, bg30 * NOISE_MARGIN))
        th15.append(max(HARD_FLOOR_15K, bg15 * NOISE_MARGIN))
    return max(th30), max(th15)


THRESHOLD_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "s3_thresholds.json")


def phase2_apply(th_30k, th_15k):
    """
    Persist calibrated thresholds to s3_thresholds.json (git-ignored).
    RF_Stage3_CycloAudit.__init__() reads this file at startup.
    The source file is never modified -- no git conflicts.
    """
    import json
    from datetime import datetime
    print("\n" + "=" * 60)
    print("  Phase 2 -- Saving thresholds to JSON")
    print(f"  THRESHOLD_30K : {th_30k*100:.2f}%")
    print(f"  THRESHOLD_15K : {th_15k*100:.2f}%")
    print(f"  File          : {THRESHOLD_JSON}")
    print("=" * 60)

    payload = {
        "THRESHOLD_30K": round(th_30k, 6),
        "THRESHOLD_15K": round(th_15k, 6),
        "calibrated_at": datetime.now().isoformat(timespec='seconds'),
    }
    with open(THRESHOLD_JSON, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  OK -- thresholds saved.")


# =============================================================================
# Calibration report plot
# =============================================================================
def _save_report(bg_results, th_30k, th_15k):
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as plt

        n    = len(SECTORS_HZ)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, freq in zip(axes, SECTORS_HZ):
            bg = bg_results.get(freq, {})
            values = [
                bg.get('ncc_30k_max', 0) * 100,
                bg.get('ncc_15k_max', 0) * 100,
                bg.get('ncc_30k_avg', 0) * 100,
                bg.get('ncc_15k_avg', 0) * 100,
            ]
            colors = ['#EF5350', '#FF7043', '#78909C', '#90A4AE']
            labels = ['BG 30kHz max', 'BG 15kHz max',
                      'BG 30kHz avg', 'BG 15kHz avg']

            bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
            ax.axhline(th_30k * 100, color='#1565C0', linestyle='--',
                       linewidth=1.5, label=f'TH_30k={th_30k*100:.1f}%')
            ax.axhline(th_15k * 100, color='#2E7D32', linestyle='--',
                       linewidth=1.5, label=f'TH_15k={th_15k*100:.1f}%')

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.1, f'{val:.2f}%',
                            ha='center', va='bottom', fontsize=9)

            ax.set_title(f'{freq/1e6:.0f} MHz Sector')
            ax.set_ylabel('CAF-NCC (%)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')
            ax.tick_params(axis='x', labelrotation=15, labelsize=8)

        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.suptitle(
            f'S3 CAF-FFT Calibration Report | '
            f'TH_30K={th_30k*100:.2f}%  TH_15K={th_15k*100:.2f}%',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f's3_calibration_{ts}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Report saved: {path}")
    except Exception as e:
        print(f"  [!] Report plot failed (non-critical): {e}")


# =============================================================================
# Main (fully automatic -- no user input required)
# =============================================================================
def main():
    print()
    print("=" * 62)
    print("  RF-Vision S3 CAF-FFT Auto Calibration v1.1")
    print("  Ambient NCC floor -> Optimal thresholds -> Auto-patch")
    print("=" * 62)
    print(f"  SDR    : {SDR_URI}")
    print(f"  Fs     : {SAMPLE_RATE/1e6:.0f} MSps")
    print(f"  Sectors: {[int(f/1e6) for f in SECTORS_HZ]} MHz")
    print()

    # Phase 1: background measurement
    bg_results = phase1_background()

    # Phase 2: derive thresholds
    th_30k, th_15k = _derive_thresholds(bg_results)

    print(f"\n  +----------------------------------+")
    print(f"  |  Derived thresholds (worst-case) |")
    print(f"  |  THRESHOLD_30K = {th_30k*100:6.2f}%          |")
    print(f"  |  THRESHOLD_15K = {th_15k*100:6.2f}%          |")
    print(f"  +----------------------------------+")

    # Auto-write (no confirmation prompt)
    phase2_apply(th_30k, th_15k)
    _save_report(bg_results, th_30k, th_15k)
    print("\n  Calibration complete. New thresholds are active.\n")


if __name__ == '__main__':
    main()
