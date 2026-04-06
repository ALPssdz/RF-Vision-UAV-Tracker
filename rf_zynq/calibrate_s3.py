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

# -- SDR parameters (read from config.py -- same values used at runtime) ------
# IMPORTANT: calibration gain MUST equal operational gain (config.SDR_GAIN_DB).
# Loading from config ensures a single source of truth.
from backend_rk3588.config import SDR_URI, SAMPLE_RATE, SDR_GAIN_DB
RX_GAIN     = SDR_GAIN_DB
BUFFER_SIZE = 2_621_440   # 65 ms @ 40 MSps
SECTORS_HZ  = [5745e6, 5785e6, 5825e6]
N_CAPTURES  = 8           # IQ captures per sector (8 frames -> more stable bg estimate)

# -- CAF scan parameters (identical to RF_Stage3_CycloAudit) ------------------
# CHUNK_SIZE = 200_000 (5 ms @ 40 MSps).
# Larger chunks (400k) were found to amplify SMPS harmonic NCC because the
# integration window covers more exact SMPS cycles -> REVERTED to 200k.
CHUNK_SIZE       = 200_000
TAU_30K, TAU_15K = 1333, 2667
ALPHA_SCAN_30K   = (22_000.0, 30_000.0)   # narrowed to OcuSync CP range
ALPHA_SCAN_15K   = (10_500.0, 14_500.0)
MIN_POWER_GATE   = 1e-5

# -- Threshold derivation parameters ------------------------------------------
# Formula: bg_eff = BG_MAX_WEIGHT*max + (1-BG_MAX_WEIGHT)*avg
#          th = max(HARD_FLOOR, bg_eff * NOISE_MARGIN)
#
# Rationale for weighted estimate:
#   Using raw max is dominated by single outlier bursts (SMPS spikes), which
#   can be 3-4x above the typical NCC level.  A weighted average (0.4*max +
#   0.6*avg) reduces outlier influence while retaining sensitivity to
#   sustained elevated backgrounds.
#
# NOISE_MARGIN = 2.0: the primary false-alarm guard is NOT the NCC threshold
# alone, but the Level 3 PSR test (peak-to-sidelobe ratio of the tau-domain
# CAF) and Level 4 CFS test (cyclic-frequency concentration).
# True OcuSync: PSR >> 2.5, CFS >> 1.8 -> passes both gates.
# SMPS burst: PSR ~ 1.0-1.5, CFS ~ 1.0-1.2 -> rejected by both gates.
# This allows NOISE_MARGIN to be set tightly (2.0x) while keeping overall
# system Pfa < 0.1% via the multi-stage funnel.
NOISE_MARGIN   = 2.0
BG_MAX_WEIGHT  = 0.4    # weight on bg_max;  (1-0.4)=0.6 weight on bg_avg
HARD_FLOOR_30K = 0.018  # 1.8%  (8x theoretical NCC floor 1/sqrt(200000))
HARD_FLOOR_15K = 0.014  # 1.4%  (6x theoretical NCC floor)

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
    Derive per-sector thresholds using an outlier-robust background estimate.

    Formula:
      bg_eff = BG_MAX_WEIGHT * bg_max + (1 - BG_MAX_WEIGHT) * bg_avg
      th     = max(HARD_FLOOR, bg_eff * NOISE_MARGIN)

    Statistical rationale:
      Raw max is dominated by single burst events (SMPS spikes, multipath).
      A weighted average (40% max + 60% avg) reduces outlier influence while
      still accounting for occasional elevated backgrounds.
      NOISE_MARGIN=3 on this tighter estimate provides the same guard as ~5x
      on the raw max for typical ambient conditions.

    Returns
    -------
    dict: {freq_hz: {'th_30k': float, 'th_15k': float}}
    """
    per_sector = {}
    for freq in SECTORS_HZ:
        bg    = bg_results.get(freq, {})
        # Weighted background estimates
        max30 = bg.get('ncc_30k_max', HARD_FLOOR_30K / NOISE_MARGIN)
        avg30 = bg.get('ncc_30k_avg', max30 * 0.5)
        max15 = bg.get('ncc_15k_max', HARD_FLOOR_15K / NOISE_MARGIN)
        avg15 = bg.get('ncc_15k_avg', max15 * 0.5)

        bg_eff_30 = BG_MAX_WEIGHT * max30 + (1 - BG_MAX_WEIGHT) * avg30
        bg_eff_15 = BG_MAX_WEIGHT * max15 + (1 - BG_MAX_WEIGHT) * avg15

        per_sector[freq] = {
            'th_30k': max(HARD_FLOOR_30K, bg_eff_30 * NOISE_MARGIN),
            'th_15k': max(HARD_FLOOR_15K, bg_eff_15 * NOISE_MARGIN),
        }
    return per_sector


THRESHOLD_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "s3_thresholds.json")


def phase2_apply(per_sector_th):
    """
    Persist per-sector calibrated thresholds to s3_thresholds.json.

    JSON schema:
      {
        "sectors": {
          "5745000000": {"th_30k": 0.050, "th_15k": 0.030},
          ...
        },
        "calibrated_at": "..."
      }
    RF_Stage3_CycloAudit.__init__() reads this file at startup.
    """
    import json
    from datetime import datetime
    print("\n" + "=" * 60)
    print("  Phase 2 -- Saving per-sector thresholds to JSON")
    for freq, th in per_sector_th.items():
        print(f"  {freq/1e6:.0f} MHz:  "
              f"TH_30k={th['th_30k']*100:.2f}%   TH_15k={th['th_15k']*100:.2f}%")
    print(f"  File : {THRESHOLD_JSON}")
    print("=" * 60)

    payload = {
        "sectors": {
            str(int(freq)): {
                "th_30k": round(th["th_30k"], 6),
                "th_15k": round(th["th_15k"], 6),
            }
            for freq, th in per_sector_th.items()
        },
        "calibrated_at": datetime.now().isoformat(timespec='seconds'),
    }
    with open(THRESHOLD_JSON, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  OK -- thresholds saved.")


# =============================================================================
# Calibration report plot
# =============================================================================
def _save_report(bg_results, per_sector_th):
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
            th = per_sector_th.get(freq, {'th_30k': HARD_FLOOR_30K, 'th_15k': HARD_FLOOR_15K})
            th_30k = th['th_30k']
            th_15k = th['th_15k']
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
            f'S3 CAF-FFT Calibration Report  (NOISE_MARGIN={NOISE_MARGIN}x)',
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

    # Phase 2: derive per-sector thresholds
    per_sector_th = _derive_thresholds(bg_results)

    print("\n  +" + "-" * 46 + "+")
    print(f"  | Per-sector derived thresholds  (NOISE_MARGIN = {NOISE_MARGIN}x)   |")
    for freq, th in per_sector_th.items():
        line = (f"  |  {freq/1e6:.0f} MHz:  "
                f"TH_30k={th['th_30k']*100:5.2f}%   "
                f"TH_15k={th['th_15k']*100:5.2f}%")
        print(line.ljust(48) + "|")
    print("  +" + "-" * 46 + "+")

    # Auto-write
    phase2_apply(per_sector_th)
    _save_report(bg_results, per_sector_th)
    print("\n  Calibration complete. New thresholds are active.\n")


if __name__ == '__main__':
    main()
