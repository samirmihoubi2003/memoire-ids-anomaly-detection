import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# paths
input_path = "data/processed/Monday_clean.csv"
output_features = "data/processed/X_monday_scaled.csv"
output_labels = "data/processed/y_monday.csv"

# create folder if needed
os.makedirs("data/processed", exist_ok=True)

# load cleaned dataset
df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)

# separate features and label
X = df.drop(" Label", axis=1)
y = df[" Label"]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert back to dataframe
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# save
X_scaled.to_csv(output_features, index=False)
y.to_csv(output_labels, index=False)

print("Preprocessed features saved to:", output_features)
print("Labels saved to:", output_labels)