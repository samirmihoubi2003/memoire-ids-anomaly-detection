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
from sklearn.metrics import classification_report

# =========================================================
# 1) Reproducibility
# =========================================================

RANDOM_STATE = 42
FINAL_PERCENTILE = 90

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


# =========================================================
# 4) Build anomaly detection training data
#    Train ONLY on BENIGN
# =========================================================

monday_benign = monday[monday[" Label"] == "BENIGN"]
tuesday_benign = tuesday[tuesday[" Label"] == "BENIGN"]
wednesday_benign = wednesday[wednesday[" Label"] == "BENIGN"]

train_df = pd.concat([monday_benign, tuesday_benign, wednesday_benign], axis=0)

test_df = friday.copy()

print("Train BENIGN dataset shape:", train_df.shape)
print("Friday test shape:", test_df.shape)

print("\nTrain label distribution:")
print(train_df[" Label"].value_counts())

print("\nFriday test label distribution:")
print(test_df[" Label"].value_counts())


# =========================================================
# 5) Split X / y
# =========================================================

X = train_df.drop(" Label", axis=1)

X_test = test_df.drop(" Label", axis=1)
y_test = test_df[" Label"].apply(lambda x: 0 if x == "BENIGN" else 1).values


# =========================================================
# 6) Train / Validation split
#    Validation also BENIGN only
# =========================================================

X_train, X_val = train_test_split(
    X, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

print("\nX_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)


# =========================================================
# 7) Log transform
#    Helps with very large CICIDS2017 feature values
# =========================================================


def signed_log1p(data):
    data = np.asarray(data, dtype=np.float32)
    return np.sign(data) * np.log1p(np.abs(data))


X_train = signed_log1p(X_train)
X_val = signed_log1p(X_val)
X_test = signed_log1p(X_test)


# =========================================================
# 8) Feature selection + scaling
#    Fit ONLY on training data
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
# 9) Convert to tensors
# =========================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


generator = torch.Generator()
generator.manual_seed(RANDOM_STATE)

train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=512,
    shuffle=True,
    generator=generator,
    num_workers=0,
)


# =========================================================
# 10) Denoising Autoencoder
# =========================================================


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


model = DenoisingAutoencoder(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================================================
# 11) Training
# =========================================================

epochs = 30
noise_std = 0.02

print("\nTraining Denoising Autoencoder...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x_clean = batch[0].to(device)

        noise = noise_std * torch.randn_like(x_clean)
        x_noisy = x_clean + noise

        optimizer.zero_grad()

        x_recon = model(x_noisy)
        loss = criterion(x_recon, x_clean)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")


# =========================================================
# 12) Threshold from BENIGN validation only
# =========================================================


def reconstruction_errors(model, X_tensor, batch_size=4096):
    model.eval()
    errors_list = []

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            x_recon = model(x)
            errors = torch.mean((x - x_recon) ** 2, dim=1)
            errors_list.append(errors.cpu().numpy())

    return np.concatenate(errors_list)


val_errors = reconstruction_errors(model, X_val_tensor)

threshold_candidates = {
    "p85": np.percentile(val_errors, 85),
    "p90": np.percentile(val_errors, 90),
    "p95": np.percentile(val_errors, 95),
    "p97": np.percentile(val_errors, 97),
    "p99": np.percentile(val_errors, 99),
}

print("\nThreshold candidates from BENIGN validation:")
for name, value in threshold_candidates.items():
    print(name, "=", value)

threshold = np.percentile(val_errors, FINAL_PERCENTILE)

print("\nSelected threshold percentile:", FINAL_PERCENTILE)
print("Selected threshold:", threshold)


# =========================================================
# 13) Final strict test on Friday
# =========================================================

test_errors = reconstruction_errors(model, X_test_tensor)

y_pred = (test_errors > threshold).astype(int)

print("\n===== Friday DDoS STRICT TEST =====")
print("Used threshold:", threshold)
print(classification_report(y_test, y_pred))


# =========================================================
# 14) Save model and preprocessing objects
# =========================================================

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/anomaly_denoising_autoencoder_best.pth")
joblib.dump(selector, "models/anomaly_dae_selector.joblib")
joblib.dump(scaler, "models/anomaly_dae_scaler.joblib")
np.save("models/anomaly_dae_threshold.npy", np.array(threshold))

print("\nSaved best anomaly detection model, selector, scaler, and threshold.")
