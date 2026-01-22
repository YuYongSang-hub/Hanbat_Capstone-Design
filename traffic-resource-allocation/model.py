# model.py
import torch
import torch.nn as nn
import config

class TrafficPredictorLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.NUM_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, N, T, F)
        B, N, T, F = x.shape
        x = x.view(B * N, T, F)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out.view(B, N)
