import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Load cleaned datasets
monday = pd.read_csv("data/processed/Monday_clean.csv")
tuesday = pd.read_csv("data/processed/Tuesday_clean.csv")

# Extract BENIGN only
monday_benign = monday[monday[" Label"] == "BENIGN"]
tuesday_benign = tuesday[tuesday[" Label"] == "BENIGN"]

print("Monday BENIGN:", monday_benign.shape)
print("Tuesday BENIGN:", tuesday_benign.shape)

# Combine
train_benign = pd.concat([monday_benign, tuesday_benign], axis=0)

print("Combined training shape:", train_benign.shape)

# Save
train_benign.to_csv("data/processed/Train_BENIGN_MultiDay.csv", index=False)

print("Saved multi-day BENIGN dataset.")
