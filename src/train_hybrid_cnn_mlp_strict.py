import os
import random
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, f1_score, fbeta_score

# =========================================================
# 1) Reproducibility
# =========================================================

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.5

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# 2) Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 3) Load datasets
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
# 4) Split X / y
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
# 5) Train / Validation split
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain split distribution:")
print(y_train.value_counts())

print("\nValidation split distribution:")
print(y_val.value_counts())


# =========================================================
# 6) Feature selection + scaling
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

input_dim = X_train.shape[1]

print("\nSelected features:", input_dim)


# =========================================================
# 7) Convert to tensors
#    CNN input shape: (samples, channels, features)
# =========================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

y_val_np = y_val.values
y_test_np = y_test.values


# =========================================================
# 8) Deterministic DataLoader
# =========================================================

generator = torch.Generator()
generator.manual_seed(RANDOM_STATE)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=1024,
    shuffle=True,
    generator=generator,
    num_workers=0,
)


# =========================================================
# 9) Hybrid CNN + MLP Classifier
# =========================================================


class HybridCNNMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # CNN branch
        self.cnn_branch = nn.Sequential(
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
            nn.Flatten(),
        )

        # MLP branch
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        cnn_features = self.cnn_branch(x)

        flat_x = x.squeeze(1)
        mlp_features = self.mlp_branch(flat_x)

        combined = torch.cat([cnn_features, mlp_features], dim=1)

        logits = self.classifier(combined)

        return logits


model = HybridCNNMLP(input_dim=input_dim).to(device)


# =========================================================
# 10) Loss function with class imbalance handling
# =========================================================

num_normal = (y_train == 0).sum()
num_attack = (y_train == 1).sum()

pos_weight = torch.tensor([num_normal / num_attack], dtype=torch.float32).to(device)

print("\nPos weight:", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================================================
# 11) Training
# =========================================================

epochs = 15

print("\nTraining Hybrid CNN + MLP Classifier...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")


# =========================================================
# 12) Validation check with fixed threshold
# =========================================================

model.eval()

with torch.no_grad():
    val_logits = model(X_val_tensor.to(device))
    val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

val_pred = (val_probs >= FINAL_THRESHOLD).astype(int)

val_f1 = f1_score(y_val_np, val_pred, zero_division=0)
val_f2 = fbeta_score(y_val_np, val_pred, beta=2, zero_division=0)

print("\n===== Validation check =====")
print("Fixed threshold:", FINAL_THRESHOLD)
print(f"Validation F1_attack: {val_f1:.4f}")
print(f"Validation F2_attack: {val_f2:.4f}")


# =========================================================
# 13) Final strict test on Friday
# =========================================================

with torch.no_grad():
    test_logits = model(X_test_tensor.to(device))
    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()

y_pred = (test_probs >= FINAL_THRESHOLD).astype(int)

print("\n===== Friday STRICT TEST =====")
print("Used threshold:", FINAL_THRESHOLD)
print(classification_report(y_test_np, y_pred))


# =========================================================
# 14) Save model and preprocessing objects
# =========================================================

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/hybrid_cnn_mlp_strict.pth")
joblib.dump(selector, "models/hybrid_cnn_mlp_selector.joblib")
joblib.dump(scaler, "models/hybrid_cnn_mlp_scaler.joblib")
np.save("models/hybrid_cnn_mlp_threshold.npy", np.array(FINAL_THRESHOLD))

print("\nSaved Hybrid CNN + MLP model, selector, scaler, and threshold.")
