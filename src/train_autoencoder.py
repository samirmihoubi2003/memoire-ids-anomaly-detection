import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
X_monday = pd.read_csv("data/processed/X_train_monday_scaled.csv")
X_test = pd.read_csv("data/processed/X_test_wednesday_scaled.csv")
y_test = pd.read_csv("data/processed/y_test_wednesday.csv")[" Label"]

# Convert labels: BENIGN = 0, Attack = 1
y_test = y_test.apply(lambda x: 0 if x == "BENIGN" else 1).values

# Split Monday into train and validation
X_train, X_val = train_test_split(
    X_monday,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=256,
    shuffle=True
)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

input_dim = X_train.shape[1]
model = Autoencoder(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20

print("Training Autoencoder with validation split...")

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

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f}")

# Compute validation reconstruction errors
model.eval()

with torch.no_grad():
    val_reconstructed = model(X_val_tensor)
    val_errors = torch.mean((X_val_tensor - val_reconstructed) ** 2, dim=1).numpy()

# Try different thresholds from validation normal data
thresholds = {
    "p90": np.percentile(val_errors, 90),
    "p95": np.percentile(val_errors, 95),
    "p97": np.percentile(val_errors, 97),
    "p99": np.percentile(val_errors, 99),
}

print("\nThreshold candidates:")
for name, value in thresholds.items():
    print(name, "=", value)

# Choose threshold
threshold = thresholds["p90"]
print("\nSelected threshold:", threshold)

# Test on Wednesday
with torch.no_grad():
    test_reconstructed = model(X_test_tensor)
    test_errors = torch.mean((X_test_tensor - test_reconstructed) ** 2, dim=1).numpy()

y_pred = (test_errors > threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))