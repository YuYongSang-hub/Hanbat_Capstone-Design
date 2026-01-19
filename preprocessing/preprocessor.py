# preprocessor.py
import numpy as np
import pandas as pd
import config
import os

def create_dataset_from_csv():
    print("Starting Preprocessing...")
    
    # 1. Load Raw Data
    df = pd.read_csv(config.RAW_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Pivot (Time x Nodes)
    # 결측치는 0으로 채움
    pivot_df = df.pivot(index="timestamp", columns="node_id", values="bytes").fillna(0)
    
    # 노드 순서 보장 (node_0, node_1, ...)
    sorted_cols = sorted(pivot_df.columns, key=lambda x: int(x.split('_')[1]))
    raw_values = pivot_df[sorted_cols].values # Shape: (Total_steps, N)
    
    timestamps = pivot_df.index.values

    # 3. Data Scaling (Log Scaling as per Spec)
    # Log(x + 1) 적용하여 데이터 범위를 줄임
    scaled_values = np.log1p(raw_values)
    
    # 4. Sliding Window
    X_list = []
    Y_list = []
    
    # (Total - Window - Horizon) 만큼 반복
    limit = len(scaled_values) - config.WINDOW_SIZE - config.PRED_HORIZON + 1
    
    for i in range(limit):
        # Input: 과거 Window Size 만큼
        window = scaled_values[i : i + config.WINDOW_SIZE]
        X_list.append(window)
        
        # Target: 바로 다음 시점의 트래픽 양 (Regression Target)
        # Note: 실제 모델 학습 시 Y_alloc(비율)이 필요하면 Softmax를 적용해 사용
        target = scaled_values[i + config.WINDOW_SIZE] 
        Y_list.append(target)

    X = np.array(X_list) # Shape: (Samples, T, N)
    Y = np.array(Y_list) # Shape: (Samples, N)
    
    # 5. Dimension Expansion: (Samples, T, N) -> (Samples, N, T, F)
    # Transpose to (Samples, N, T)
    X = np.transpose(X, (0, 2, 1))
    # Add Feature Dimension
    X = X[..., np.newaxis] 
    
    print(f"Tensor Created - X: {X.shape}, Y: {Y.shape}")
    
    # 6. Chronological Split (No Shuffle)
    split_idx = int(len(X) * config.TRAIN_RATIO)
    
    x_train, x_test = X[:split_idx], X[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]
    
    # 7. Save to .npz
    save_dir = config.PROCESSED_DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez(f"{save_dir}/traffic_data_train.npz", x_data=x_train, y_data=y_train)
    np.savez(f"{save_dir}/traffic_data_test.npz", x_data=x_test, y_data=y_test)
    
    print(f"✅ Processing Complete!")
    print(f"   Train: {x_train.shape}")
    print(f"   Test:  {x_test.shape}")
    print(f"   Saved to: {save_dir}")

if __name__ == "__main__":
    create_dataset_from_csv()
