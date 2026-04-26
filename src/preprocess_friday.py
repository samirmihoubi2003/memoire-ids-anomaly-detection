import pandas as pd
import joblib

# Load Friday clean data
df = pd.read_csv("data/processed/Friday_DDoS_clean.csv")

# Split X and y
X = df.drop(" Label", axis=1)
y = df[" Label"]

print("Friday shape:", X.shape)

# Load selector + scaler (من Monday فقط)
selector = joblib.load("data/processed/variance_selector_monday.joblib")
scaler = joblib.load("data/processed/standard_scaler_selected_monday.joblib")

# Apply feature selection
X_selected = selector.transform(X)

# Convert to DataFrame
selected_columns = selector.get_support()
X_selected = pd.DataFrame(X_selected)

# Apply scaling
X_scaled = scaler.transform(X_selected)
X_scaled = pd.DataFrame(X_scaled)

# Save
X_scaled.to_csv("data/processed/X_test_friday_scaled.csv", index=False)
y.to_csv("data/processed/y_test_friday.csv", index=False)

print("Friday preprocessing done.")
