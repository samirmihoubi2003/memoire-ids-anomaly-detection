import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Load datasets (الموجودة عندك فقط)
monday = pd.read_csv("data/processed/Monday_clean.csv")
tuesday = pd.read_csv("data/processed/Tuesday_clean.csv")
wednesday = pd.read_csv("data/processed/Wednesday_clean.csv")
friday = pd.read_csv("data/processed/Friday_DDoS_clean.csv")

# Extract BENIGN
monday_b = monday[monday[" Label"] == "BENIGN"]
tuesday_b = tuesday[tuesday[" Label"] == "BENIGN"]
wednesday_b = wednesday[wednesday[" Label"] == "BENIGN"]
friday_b = friday[friday[" Label"] == "BENIGN"]

print("Monday:", monday_b.shape)
print("Tuesday:", tuesday_b.shape)
print("Wednesday:", wednesday_b.shape)
print("Friday:", friday_b.shape)

# Combine all
train_all = pd.concat([monday_b, tuesday_b, wednesday_b, friday_b], axis=0)

print("Final training shape:", train_all.shape)

# Save
train_all.to_csv("data/processed/Train_BENIGN_AllDays.csv", index=False)

print("Saved ALL BENIGN dataset.")
