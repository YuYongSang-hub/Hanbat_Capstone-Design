# generator.py
import numpy as np
import pandas as pd
import config
import os

def generate_synthetic_traffic():
    """
    명세서의 [트래픽 설계 원칙]을 반영한 가상 데이터를 생성하여 CSV로 저장합니다.
    - 시간적 변동성: Sine wave + Random Noise
    - 노드 불균형: 특정 노드에 가중치 부여
    - Burst: 간헐적인 트래픽 폭주
    """
    np.random.seed(config.RANDOM_SEED)
    
    # 1. 시간축 생성
    timestamps = pd.date_range(
        start="2026-01-19 09:00:00", 
        periods=config.TOTAL_SAMPLES, 
        freq=f"{config.INTERVAL_SEC}s"
    )
    
    data = []
    
    # 기본 패턴 (Sine Wave)
    time_idx = np.arange(config.TOTAL_SAMPLES)
    base_pattern = 1000 * (np.sin(time_idx * 0.01) + 1) + 500  # Base traffic
    
    print(f"Generating traffic for {config.NUM_NODES} nodes...")
    
    for node_idx in range(config.NUM_NODES):
        node_id = f"node_{node_idx}"
        
        # 2. 노드별 특성 부여 (Node Imbalance)
        # 짝수 노드는 트래픽이 많고, 홀수 노드는 적게 설정
        usage_factor = 2.0 if node_idx % 2 == 0 else 0.5
        
        # 3. 노이즈 및 패턴 추가
        noise = np.random.normal(0, 100, size=config.TOTAL_SAMPLES)
        traffic = (base_pattern * usage_factor) + noise
        
        # 4. Burst (간헐적 폭주) 추가 - 약 1% 확률
        burst_indices = np.random.choice(config.TOTAL_SAMPLES, size=int(config.TOTAL_SAMPLES * 0.01), replace=False)
        traffic[burst_indices] *= 5.0  # 평소의 5배 폭주
        
        # 음수 제거 (트래픽은 0 이상)
        traffic = np.maximum(traffic, 0).astype(int)
        
        # 데이터프레임용 리스트 추가
        for ts, bytes_val in zip(timestamps, traffic):
            data.append([ts, node_id, bytes_val])

    # DataFrame 생성 및 저장
    df = pd.DataFrame(data, columns=["timestamp", "node_id", "bytes"])
    
    # 저장 경로 확보
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"✅ Raw data generated at: {config.RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_traffic()
