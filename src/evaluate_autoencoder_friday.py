import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report

# Load Friday data
X_test = pd.read_csv("data/processed/X_test_friday_scaled.csv")
y_test = pd.read_csv("data/processed/y_test_friday.csv")[" Label"]

# Convert labels
y_test = y_test.apply(lambda x: 0 if x == "BENIGN" else 1).values

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)


# Same model
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
        return self.decoder(self.encoder(x))


# Load model
input_dim = X_test.shape[1]
model = Autoencoder(input_dim)
model.load_state_dict(torch.load("models/autoencoder_model.pth"))
model.eval()

# Load threshold
threshold = np.load("models/autoencoder_threshold.npy")

print("Loaded threshold:", threshold)

# Compute reconstruction error
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()

# Predict
y_pred = (errors > threshold).astype(int)

# Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
