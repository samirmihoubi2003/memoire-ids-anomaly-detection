import pandas as pd
import os
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

os.makedirs("data/processed", exist_ok=True)

# Load training data
train = pd.read_csv("data/processed/Train_BENIGN_MultiDay.csv")

# Load test sets
wednesday = pd.read_csv("data/processed/Wednesday_clean.csv")
friday = pd.read_csv("data/processed/Friday_DDoS_clean.csv")

# Split
X_train = train.drop(" Label", axis=1)

X_wed = wednesday.drop(" Label", axis=1)
y_wed = wednesday[" Label"]

X_fri = friday.drop(" Label", axis=1)
y_fri = friday[" Label"]

print("Train shape:", X_train.shape)

# Feature Selection
selector = VarianceThreshold(threshold=0.0)
X_train_sel = selector.fit_transform(X_train)

X_wed_sel = selector.transform(X_wed)
X_fri_sel = selector.transform(X_fri)

print("Selected features:", X_train_sel.shape[1])

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_wed_scaled = scaler.transform(X_wed_sel)
X_fri_scaled = scaler.transform(X_fri_sel)

# Save
pd.DataFrame(X_train_scaled).to_csv("data/processed/X_train_multiday.csv", index=False)
pd.DataFrame(X_wed_scaled).to_csv(
    "data/processed/X_wednesday_multiday.csv", index=False
)
pd.DataFrame(X_fri_scaled).to_csv("data/processed/X_friday_multiday.csv", index=False)

y_wed.to_csv("data/processed/y_wednesday_multiday.csv", index=False)
y_fri.to_csv("data/processed/y_friday_multiday.csv", index=False)

joblib.dump(selector, "data/processed/selector_multiday.joblib")
joblib.dump(scaler, "data/processed/scaler_multiday.joblib")

print("Multiday preprocessing done.")
