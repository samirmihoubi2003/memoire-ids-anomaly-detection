import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# 1) Load data
# =========================

X_monday = pd.read_csv("data/processed/X_train_monday_selected_scaled.csv")

X_wednesday = pd.read_csv("data/processed/X_test_wednesday_selected_scaled.csv")
y_wednesday = pd.read_csv("data/processed/y_test_wednesday_selected.csv")[" Label"]

X_friday = pd.read_csv("data/processed/X_test_friday_scaled.csv")
y_friday = pd.read_csv("data/processed/y_test_friday.csv")[" Label"]

# Convert labels: BENIGN = 0, Attack = 1
y_wednesday = y_wednesday.apply(lambda x: 0 if x == "BENIGN" else 1).values
y_friday = y_friday.apply(lambda x: 0 if x == "BENIGN" else 1).values

# =========================
# 2) Split Monday into train/validation
# =========================

X_train, X_val = train_test_split(
    X_monday, test_size=0.2, random_state=42, shuffle=True
)

# =========================
# 3) Convert to CNN shape
# CNN expects: (samples, channels, features)
# =========================

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)

X_wednesday_tensor = torch.tensor(X_wednesday.values, dtype=torch.float32).unsqueeze(1)
X_friday_tensor = torch.tensor(X_friday.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)

# =========================
# 4) CNN Autoencoder
# =========================


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # If output length is slightly different, crop to input length
        x_hat = x_hat[:, :, : x.shape[2]]
        return x_hat


model = CNNAutoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20

print("Training CNN Autoencoder...")

# =========================
# 5) Train
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
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {avg_loss:.6f} - "
        f"Val Loss: {val_loss:.6f}"
    )

# =========================
# 6) Threshold from validation
# =========================

model.eval()

with torch.no_grad():
    val_reconstructed = model(X_val_tensor)
    val_errors = torch.mean((X_val_tensor - val_reconstructed) ** 2, dim=(1, 2)).numpy()

thresholds = {
    "p90": np.percentile(val_errors, 90),
    "p95": np.percentile(val_errors, 95),
    "p97": np.percentile(val_errors, 97),
    "p99": np.percentile(val_errors, 99),
}

print("\nThreshold candidates:")
for name, value in thresholds.items():
    print(name, "=", value)

threshold = thresholds["p90"]
print("\nSelected threshold:", threshold)

# =========================
# 7) Evaluation function
# =========================


def evaluate_dataset(name, X_tensor, y_true):
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

    y_pred = (errors > threshold).astype(int)

    print(f"\n===== {name} Evaluation =====")
    print(classification_report(y_true, y_pred))


# =========================
# 8) Test on Wednesday and Friday
# =========================

evaluate_dataset("Wednesday", X_wednesday_tensor, y_wednesday)
evaluate_dataset("Friday DDoS", X_friday_tensor, y_friday)
