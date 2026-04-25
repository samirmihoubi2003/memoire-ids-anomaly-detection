import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# paths
input_path = "data/processed/Wednesday_clean.csv"
output_features = "data/processed/X_wednesday_scaled.csv"
output_labels = "data/processed/y_wednesday.csv"

# create folder if needed
os.makedirs("data/processed", exist_ok=True)

# load data
df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)

# split
X = df.drop(" Label", axis=1)
y = df[" Label"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert to dataframe
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# save
X_scaled.to_csv(output_features, index=False)
y.to_csv(output_labels, index=False)

print("Saved:", output_features)
print("Saved:", output_labels)