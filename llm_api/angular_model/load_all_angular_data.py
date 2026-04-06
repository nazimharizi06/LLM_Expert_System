import os
import re
import pandas as pd
from scipy.io import loadmat

BASE_DIR = os.path.dirname(__file__)
OUTPUT_CSV = os.path.join(BASE_DIR, "angle_dataset.csv")

base_path = r"\\192.168.1.114\THz Research\CommData\Angular"

all_rows = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if not file.endswith(".mat"):
            continue

        full_path = os.path.join(root, file)
        print("Loading:", file)

        # Example filename:
        # AngularComm_16QAM_300GHz_26-26_-12deg_000.mat

        modulation_match = re.search(r'_(\d+)QAM_', file)
        angle_match = re.search(r'(-?\d+)deg', file)

        folder_name = os.path.basename(root)   # e.g. 26-46
        tx_rx = folder_name.split("-")

        if modulation_match:
            modulation_qam = int(modulation_match.group(1))
        else:
            modulation_qam = -1

        if angle_match:
            angle_deg = int(angle_match.group(1))
        else:
            angle_deg = 0

        tx_antenna = int(tx_rx[0])
        rx_antenna = int(tx_rx[1])

        mat_data = loadmat(full_path)

        # confirmed key from your test
        signal_data = mat_data["data"]
        signal = signal_data.flatten()

        # temporary proxy target
        received_power = (signal ** 2).mean()

        row = {
            "tx_angle_deg": angle_deg,
            "rx_angle_deg": angle_deg,
            "tx_antenna": tx_antenna,
            "rx_antenna": rx_antenna,
            "modulation_qam": modulation_qam,
            "received_power": received_power,
            "file_name": file
        }

        all_rows.append(row)

df = pd.DataFrame(all_rows)

print("\nFinal dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)

df.to_csv("angle_dataset.csv", index=False)

print("\nSaved combined dataset as angle_dataset.csv")
