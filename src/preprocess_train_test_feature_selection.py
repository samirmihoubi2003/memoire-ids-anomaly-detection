import pandas as pd
import os
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Paths
monday_path = "data/processed/Monday_clean.csv"
wednesday_path = "data/processed/Wednesday_clean.csv"

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Load cleaned data
monday = pd.read_csv(monday_path)
wednesday = pd.read_csv(wednesday_path)

print("Monday shape:", monday.shape)
print("Wednesday shape:", wednesday.shape)

# Split X and y
X_train = monday.drop(" Label", axis=1)
y_train = monday[" Label"]

X_test = wednesday.drop(" Label", axis=1)
y_test = wednesday[" Label"]

print("Original number of features:", X_train.shape[1])

# Feature Selection: fit ONLY on Monday
selector = VarianceThreshold(threshold=0.0)
X_train_selected = selector.fit_transform(X_train)

# Apply same selected features to Wednesday
X_test_selected = selector.transform(X_test)

selected_columns = X_train.columns[selector.get_support()]

print("Selected number of features:", len(selected_columns))
print("Removed number of features:", X_train.shape[1] - len(selected_columns))

# Convert back to DataFrame
X_train_selected = pd.DataFrame(X_train_selected, columns=selected_columns)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_columns)

# Scaling: fit ONLY on Monday selected features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_columns)

# Save new files
X_train_scaled.to_csv(f"{output_dir}/X_train_monday_selected_scaled.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train_monday_selected.csv", index=False)

X_test_scaled.to_csv(f"{output_dir}/X_test_wednesday_selected_scaled.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test_wednesday_selected.csv", index=False)

# Save selector and scaler
joblib.dump(selector, f"{output_dir}/variance_selector_monday.joblib")
joblib.dump(scaler, f"{output_dir}/standard_scaler_selected_monday.joblib")

print("Saved selected and scaled train/test files.")
