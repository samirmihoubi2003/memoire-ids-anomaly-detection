import pandas as pd
import numpy as np
import os

input_path = "data/raw/Tuesday-WorkingHours.pcap_ISCX.csv"
output_path = "data/processed/Tuesday_clean.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(input_path)

print("Original shape:", df.shape)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean = df.dropna()

print("Shape after cleaning:", df_clean.shape)

df_clean.to_csv(output_path, index=False)

print("Cleaned dataset saved to:", output_path)

print("\nLabel distribution:")
print(df_clean[" Label"].value_counts())
