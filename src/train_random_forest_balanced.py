import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load Wednesday test data prepared with leakage-free scaler
X = pd.read_csv("data/processed/X_test_wednesday_scaled.csv")
y_raw = pd.read_csv("data/processed/y_test_wednesday.csv")[" Label"]

# Convert labels:
# BENIGN = 0
# any attack = 1
y = y_raw.apply(lambda label: 0 if label == "BENIGN" else 1)

# Separate normal and attack samples
normal_idx = y[y == 0].index
attack_idx = y[y == 1].index

# Balanced training size
n_train_per_class = 20000

# Sample the same number from each class for training
normal_train_idx = normal_idx.to_series().sample(
    n=n_train_per_class, random_state=42
).index

attack_train_idx = attack_idx.to_series().sample(
    n=n_train_per_class, random_state=42
).index

train_idx = normal_train_idx.union(attack_train_idx)

# Everything not used for training remains for testing
test_idx = y.index.difference(train_idx)

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

print("Training distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

# Train supervised balanced baseline
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("Testing Random Forest...")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))