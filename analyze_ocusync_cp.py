import os
import numpy as np
from scipy.signal import resample_poly

def run_analysis():
    dataset_path = r"e:\Myprojects\RF-Vision-UAV-Tracker\Drone RF Data\DJI MINI4 PRO\VTSBW=20\pack2_0-1s.iq"
    
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    print("1. Loading raw IQ data (100MSPS)...")
    raw_iq = np.fromfile(dataset_path, dtype=np.complex64, count=5000000) # 50ms data
    
    print("2. Resampling to 40MSPS (to match SDR receiver)...")
    resampled_iq = resample_poly(raw_iq, up=2, down=5)
    
    normalized_iq = resampled_iq / np.max(np.abs(resampled_iq))
    normalized_iq = normalized_iq - np.mean(normalized_iq) # DC remove
    
    print("3. Scanning CP Correlation Delays...")
    # We scan delays from 10 to 4000 samples.
    # We are looking for delays that produce high correlation > 0.05
    
    delays = np.arange(10, 4000, 1)
    correlations = []
    
    # Fast correlation approach
    total_power = np.mean(np.abs(normalized_iq)**2)
    
    for d in delays:
        iq_main = normalized_iq[d:]
        iq_delayed = normalized_iq[:-d]
        corr = np.abs(np.mean(iq_main * np.conj(iq_delayed))) / (total_power + 1e-12)
        correlations.append(corr)

    correlations = np.array(correlations)
    
    # Find local maxima
    print("\n--- Physical Delay Signatures Found ---")
    peak_indices = np.argsort(correlations)[-10:][::-1] # top 10 delays (excluding 0)
    for idx in peak_indices:
        real_delay = delays[idx]
        corr_val = correlations[idx]
        if corr_val > 0.01:
            print(f"Delay: {real_delay:4d} samples | Correlation: {corr_val:.4f} | Est Subcarrier: {40e6/real_delay/1e3:.2f} kHz")

if __name__ == "__main__":
    run_analysis()
