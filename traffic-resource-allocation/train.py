# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config
from model import TrafficPredictorLSTM
import os

def train():
    data = np.load(f"{config.PROCESSED_DATA_DIR}/traffic_data_train.npz")
    x = torch.FloatTensor(data["x_data"])
    y = torch.FloatTensor(data["y_data"])

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    model = TrafficPredictorLSTM()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/traffic_lstm.pth")
    print("✅ Model saved")

if __name__ == "__main__":
    train()
