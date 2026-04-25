import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# load data
X_train = pd.read_csv("data/processed/X_monday_scaled.csv")
X_test = pd.read_csv("data/processed/X_wednesday_scaled.csv")
y_test = pd.read_csv("data/processed/y_wednesday.csv")

# تحويل labels إلى 0 و 1
# 0 = normal (BENIGN)
# 1 = anomaly (attack)
y_test = y_test[" Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# build model
model = IsolationForest(contamination=0.1, random_state=42)

print("Training model...")
model.fit(X_train)

print("Testing model...")
y_pred = model.predict(X_test)

# IsolationForest يعطي:
# 1 = normal
# -1 = anomaly

y_pred = [0 if x == 1 else 1 for x in y_pred]

# evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))