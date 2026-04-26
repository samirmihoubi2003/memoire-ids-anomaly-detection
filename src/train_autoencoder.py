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
X_test = pd.read_csv("data/processed/X_test_wednesday_selected_scaled.csv")
y_test = pd.read_csv("data/processed/y_test_wednesday_selected.csv")[" Label"]
# BENIGN = 0, Attack = 1
y_test = y_test.apply(lambda x: 0 if x == "BENIGN" else 1).values

# =========================
# 2) Split Monday into train/validation
# =========================

X_train, X_val = train_test_split(
    X_monday, test_size=0.2, random_state=42, shuffle=True
)

# =========================
# 3) Convert to tensors
# =========================

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)

# =========================
# 4) Denoising Autoencoder model
# =========================


class Autoencoder(nn.Module):
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


input_dim = X_train.shape[1]
model = Autoencoder(input_dim)

# =========================
# 5) Training setup
# =========================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
noise_factor = 0.05

print("Training Denoising Autoencoder...")

# =========================
# 6) Training loop
# =========================

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x = batch[0]

        # Add small noise to normal training data
        noise = noise_factor * torch.randn_like(x)
        x_noisy = x + noise

        optimizer.zero_grad()

        # Input = noisy data
        x_hat = model(x_noisy)

        # Target = clean data
        loss = criterion(x_hat, x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation is evaluated without adding noise
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
# 7) Compute validation errors
# =========================

model.eval()

with torch.no_grad():
    val_reconstructed = model(X_val_tensor)
    val_errors = torch.mean((X_val_tensor - val_reconstructed) ** 2, dim=1).numpy()

thresholds = {
    "p90": np.percentile(val_errors, 90),
    "p95": np.percentile(val_errors, 95),
    "p97": np.percentile(val_errors, 97),
    "p99": np.percentile(val_errors, 99),
}

print("\nThreshold candidates:")
for name, value in thresholds.items():
    print(name, "=", value)

# Current selected threshold
threshold = thresholds["p90"]

print("\nSelected threshold:", threshold)

# =========================
# 8) Test on Wednesday
# =========================

with torch.no_grad():
    test_reconstructed = model(X_test_tensor)
    test_errors = torch.mean((X_test_tensor - test_reconstructed) ** 2, dim=1).numpy()

y_pred = (test_errors > threshold).astype(int)

# =========================
# 9) Evaluation
# =========================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
