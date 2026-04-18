import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "angle_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "angle_model.pkl")

# Load the combined angular dataset
df = pd.read_csv(DATA_PATH)

print("Dataset preview:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Input features
X = df[[
    "G_tx_db",
    "G_rx_db",
    "angle_deg",
    "modulation_qam"
]]

# Target
y = df["received_power"]

# Create and train model
knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn.fit(X, y)

# Save model
joblib.dump(knn, MODEL_PATH)

print(f"Angle model trained and saved to {MODEL_PATH}.")