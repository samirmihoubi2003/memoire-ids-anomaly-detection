import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# 1) Load data
X_train = pd.read_csv("data/processed/X_train_monday_scaled.csv")
X_test = pd.read_csv("data/processed/X_test_wednesday_scaled.csv")
y_test = pd.read_csv("data/processed/y_test_wednesday.csv")[" Label"]

# 2) Convert labels: BENIGN = 0, Attack = 1
y_test = y_test.apply(lambda x: 0 if x == "BENIGN" else 1).values

# 3) Convert data to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# 4) DataLoader
train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=256,
    shuffle=True
)

# 5) Autoencoder model
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

# 6) Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7) Train
epochs = 20

print("Training Autoencoder...")

for epoch in range(epochs):
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
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

# 8) Compute reconstruction error on training data
model.eval()

with torch.no_grad():
    train_reconstructed = model(X_train_tensor)
    train_errors = torch.mean((X_train_tensor - train_reconstructed) ** 2, dim=1).numpy()

# 9) Threshold from Monday normal data
threshold = np.percentile(train_errors, 95)

print("Threshold:", threshold)

# 10) Test on Wednesday
with torch.no_grad():
    test_reconstructed = model(X_test_tensor)
    test_errors = torch.mean((X_test_tensor - test_reconstructed) ** 2, dim=1).numpy()

# 11) Predict
y_pred = (test_errors > threshold).astype(int)

# 12) Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))