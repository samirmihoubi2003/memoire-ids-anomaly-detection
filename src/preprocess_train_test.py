import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

# paths
monday_path = "data/processed/Monday_clean.csv"
wednesday_path = "data/processed/Wednesday_clean.csv"

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# load data
monday = pd.read_csv(monday_path)
wednesday = pd.read_csv(wednesday_path)

print("Monday shape:", monday.shape)
print("Wednesday shape:", wednesday.shape)

# split features and labels
X_train = monday.drop(" Label", axis=1)
y_train = monday[" Label"]

X_test = wednesday.drop(" Label", axis=1)
y_test = wednesday[" Label"]

# fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# transform test data using the SAME scaler
X_test_scaled = scaler.transform(X_test)

# convert back to dataframes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# save files
X_train_scaled.to_csv(f"{output_dir}/X_train_monday_scaled.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train_monday.csv", index=False)

X_test_scaled.to_csv(f"{output_dir}/X_test_wednesday_scaled.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test_wednesday.csv", index=False)

# save scaler
joblib.dump(scaler, f"{output_dir}/standard_scaler_monday.joblib")

print("Saved train and test preprocessing files.")
print("Scaler fitted only on Monday and applied to Wednesday.")