import os
import re
import pandas as pd
from scipy.io import loadmat

BASE_DIR = os.path.dirname(__file__)
OUTPUT_CSV = os.path.join(BASE_DIR, "angle_dataset.csv")

base_path = r"\\192.168.1.114\THz Research\CommData\Angular"

if not os.path.exists(base_path):
    raise FileNotFoundError(
        f"NAS path not found: {base_path}\n"
        "Make sure the NAS is connected and accessible."
    )

all_rows = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if not file.lower().endswith(".mat"):
            continue

        full_path = os.path.join(root, file)
        print("Loading:", full_path)

        modulation_match = re.search(r'_(\d+)QAM_', file)
        angle_match = re.search(r'(-?\d+)deg', file)

        folder_name = os.path.basename(root)   # e.g. 26-46
        tx_rx = folder_name.split("-")

        if len(tx_rx) != 2:
            print(f"Skipping {file}: invalid folder format '{folder_name}'")
            continue

        try:
            tx_antenna_mm = int(tx_rx[0])
            rx_antenna_mm = int(tx_rx[1])
        except ValueError:
            print(f"Skipping {file}: could not parse antenna pair from folder '{folder_name}'")
            continue

        if modulation_match:
            modulation_qam = int(modulation_match.group(1))
        else:
            modulation_qam = -1

        if angle_match:
            angle_deg = int(angle_match.group(1))
        else:
            angle_deg = 0

        try:
            mat_data = loadmat(full_path)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

        if "data" not in mat_data:
            print(f"Skipping {file}: no 'data' key found")
            continue

        signal_data = mat_data["data"]
        signal = signal_data.flatten()

        if len(signal) == 0:
            print(f"Skipping {file}: empty signal")
            continue

        # waveform is in volts; assume 50-ohm RF system
        received_power = (signal ** 2).mean() / 50

        row = {
            "tx_antenna_mm": tx_antenna_mm,
            "rx_antenna_mm": rx_antenna_mm,
            "angle_deg": angle_deg,
            "modulation_qam": modulation_qam,
            "received_power": received_power,
            "file_name": file
        }

        all_rows.append(row)

if not all_rows:
    raise ValueError("No .mat files were loaded from the NAS path.")

df = pd.DataFrame(all_rows)

print("\nFinal dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved combined dataset as {OUTPUT_CSV}")
