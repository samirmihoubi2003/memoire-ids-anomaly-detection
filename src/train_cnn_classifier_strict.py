import os
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, fbeta_score, f1_score

# =========================================================
# 1) Reproducibility
# =========================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# =========================================================
# 2) Load datasets
# =========================================================

monday = pd.read_csv("data/processed/Monday_clean.csv")
tuesday = pd.read_csv("data/processed/Tuesday_clean.csv")
wednesday = pd.read_csv("data/processed/Wednesday_clean.csv")
friday = pd.read_csv("data/processed/Friday_DDoS_clean.csv")

# Strict setup:
# Train/Validation = Monday + Tuesday + Wednesday
# Final Test       = Friday only
train_df = pd.concat([monday, tuesday, wednesday], axis=0)
test_df = friday.copy()

print("Train dataset shape:", train_df.shape)
print("Friday test shape:", test_df.shape)


# =========================================================
# 3) Split X / y
# =========================================================

X = train_df.drop(" Label", axis=1)
y = train_df[" Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

X_test = test_df.drop(" Label", axis=1)
y_test = test_df[" Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

print("\nTraining label distribution:")
print(y.value_counts())

print("\nFriday test label distribution:")
print(y_test.value_counts())


# =========================================================
# 4) Train / Validation split
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain split distribution:")
print(y_train.value_counts())

print("\nValidation split distribution:")
print(y_val.value_counts())


# =========================================================
# 5) Feature selection and scaling
#    Fit ONLY on training split
# =========================================================

selector = VarianceThreshold(threshold=0.0)

X_train = selector.fit_transform(X_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nSelected features:", X_train.shape[1])


# =========================================================
# 6) Convert to CNN tensor format
#    CNN expects: (samples, channels, features)
# =========================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_np = y_val.values
y_test_np = y_test.values

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=1024, shuffle=True
)


# =========================================================
# 7) CNN Classifier
# =========================================================


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNNClassifier()


# =========================================================
# 8) Loss function with class imbalance handling
# =========================================================

num_normal = (y_train == 0).sum()
num_attack = (y_train == 1).sum()

pos_weight = torch.tensor([num_normal / num_attack], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================================================
# 9) Training
# =========================================================

epochs = 15

print("\nTraining CNN Classifier...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")


# =========================================================
# 10) Choose threshold on validation only
# =========================================================

model.eval()

with torch.no_grad():
    val_logits = model(X_val_tensor)
    val_probs = torch.sigmoid(val_logits).numpy().flatten()

threshold_candidates = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]

print("\n===== Validation threshold search =====")

best_threshold = None
best_score = -1

for threshold in threshold_candidates:
    val_pred = (val_probs >= threshold).astype(int)

    # F2 gives more importance to attack recall than precision
    score = fbeta_score(y_val_np, val_pred, beta=2, zero_division=0)
    f1 = f1_score(y_val_np, val_pred, zero_division=0)

    print(
        f"Threshold={threshold:.2f} | "
        f"F1_attack={f1:.4f} | "
        f"F2_attack={score:.4f}"
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold

print("\nSelected threshold from validation:", best_threshold)


# =========================================================
# 11) Final strict test on Friday
# =========================================================

with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits).numpy().flatten()

y_pred = (test_probs >= best_threshold).astype(int)

print("\n===== Friday STRICT TEST =====")
print(classification_report(y_test_np, y_pred))


# =========================================================
# 12) Save model and preprocessing objects
# =========================================================

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/cnn_classifier_strict.pth")
joblib.dump(selector, "models/cnn_classifier_selector.joblib")
joblib.dump(scaler, "models/cnn_classifier_scaler.joblib")
np.save("models/cnn_classifier_threshold.npy", np.array(best_threshold))

print("\nSaved CNN classifier model, selector, scaler, and threshold.")
