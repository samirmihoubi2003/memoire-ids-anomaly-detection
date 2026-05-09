import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

morning_path = "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
afternoon_path = "data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"

morning = pd.read_csv(morning_path)
afternoon = pd.read_csv(afternoon_path)

print("Morning original shape:", morning.shape)
print("Afternoon original shape:", afternoon.shape)

thursday = pd.concat([morning, afternoon], axis=0)

print("Combined original shape:", thursday.shape)

thursday.replace([np.inf, -np.inf], np.nan, inplace=True)
thursday_clean = thursday.dropna()

print("Shape after cleaning:", thursday_clean.shape)

thursday_clean.to_csv("data/processed/Thursday_clean.csv", index=False)

print("Cleaned Thursday saved to: data/processed/Thursday_clean.csv")

print("\nLabel distribution:")
print(thursday_clean[" Label"].value_counts())
