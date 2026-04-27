import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# Load data
# =========================

X_train_full = pd.read_csv("data/processed/X_train_multiday.csv")

X_wed = pd.read_csv("data/processed/X_wednesday_multiday.csv")
y_wed = pd.read_csv("data/processed/y_wednesday_multiday.csv")[" Label"]

X_fri = pd.read_csv("data/processed/X_friday_multiday.csv")
y_fri = pd.read_csv("data/processed/y_friday_multiday.csv")[" Label"]

# Convert labels
y_wed = y_wed.apply(lambda x: 0 if x == "BENIGN" else 1).values
y_fri = y_fri.apply(lambda x: 0 if x == "BENIGN" else 1).values

# =========================
# Train / Validation split
# =========================

X_train, X_val = train_test_split(
    X_train_full, test_size=0.2, random_state=42, shuffle=True
)

# =========================
# Convert to CNN format
# (samples, channels, features)
# =========================

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)

X_wed_tensor = torch.tensor(X_wed.values, dtype=torch.float32).unsqueeze(1)
X_fri_tensor = torch.tensor(X_fri.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)

# =========================
# CNN Autoencoder
# =========================


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat[:, :, : x.shape[2]]
        return x_hat


model = CNNAutoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20

print("Training CNN Autoencoder (Multi-Day)...")

# =========================
# Training
# =========================

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x = batch[0]

        optimizer.zero_grad()
        x_hat = model(x)

        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_hat = model(X_val_tensor)
        val_loss = criterion(val_hat, X_val_tensor).item()

    print(
        f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f}"
    )

# =========================
# Threshold
# =========================

model.eval()

with torch.no_grad():
    val_recon = model(X_val_tensor)
    val_errors = torch.mean((X_val_tensor - val_recon) ** 2, dim=(1, 2)).numpy()

threshold = np.percentile(val_errors, 90)

print("\nSelected threshold:", threshold)

# =========================
# Evaluation function
# =========================


def evaluate(name, X_tensor, y_true):
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).numpy()

    y_pred = (errors > threshold).astype(int)

    print(f"\n==== {name} ====")
    print(classification_report(y_true, y_pred))


# =========================
# Evaluate
# =========================

evaluate("Wednesday", X_wed_tensor, y_wed)
evaluate("Friday", X_fri_tensor, y_fri)
