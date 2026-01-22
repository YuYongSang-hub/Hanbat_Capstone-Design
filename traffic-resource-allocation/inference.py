# inference.py
import torch
import numpy as np
import config
from model import TrafficPredictorLSTM

def allocation_policy(pred, capacity):
    P = torch.softmax(pred, dim=1)
    alloc = P * capacity

    max_ratio = P.max(dim=1).values
    actions = []

    for r in max_ratio:
        if r > config.SCALE_OUT_TH:
            actions.append("SCALE_OUT")
        elif r < config.SCALE_IN_TH:
            actions.append("SCALE_IN")
        else:
            actions.append("HOLD")

    return P, alloc, actions

def run_inference():
    data = np.load(f"{config.PROCESSED_DATA_DIR}/traffic_data_test.npz")
    x_test = torch.FloatTensor(data["x_data"])
    y_test = torch.FloatTensor(data["y_data"])

    model = TrafficPredictorLSTM()
    model.load_state_dict(torch.load("models/traffic_lstm.pth"))
    model.eval()

    with torch.no_grad():
        pred = model(x_test)

    P, alloc, actions = allocation_policy(pred, config.TOTAL_CAPACITY)

    print("\nSample | MaxRatio | Decision")
    print("-" * 30)
    for i in range(5):
        print(f"{i:>3}    | {P[i].max():.3f}    | {actions[i]}")

    mse = torch.mean((pred - y_test) ** 2).item()
    print(f"\n📈 Test MSE: {mse:.4f}")
    print(f"📊 OUT:{actions.count('SCALE_OUT')} "
          f"IN:{actions.count('SCALE_IN')} "
          f"HOLD:{actions.count('HOLD')}")

if __name__ == "__main__":
    run_inference()
