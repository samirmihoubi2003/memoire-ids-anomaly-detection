import pandas as pd
import numpy as np
import os

# paths
input_path = "data/raw/Monday-WorkingHours.pcap_ISCX.csv"
output_path = "data/processed/Monday_clean.csv"

# create processed folder if not exists
os.makedirs("data/processed", exist_ok=True)

# load dataset
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# drop rows with NaN
df_clean = df.dropna()

print("Shape after cleaning:", df_clean.shape)

# save cleaned dataset
df_clean.to_csv(output_path, index=False)

print("Cleaned dataset saved to:", output_path)

# show label distribution
print("\nLabel distribution:")
print(df_clean[" Label"].value_counts())