# config.py

# --- Path Settings ---
RAW_DATA_PATH = "data/raw/traffic_log.csv"
PROCESSED_DATA_DIR = "data/processed"

# --- Global Parameters (Spec 1.2 & 3.2) ---
INTERVAL_SEC = 5       # 트래픽 샘플링 간격
WINDOW_SIZE = 12       # T: 과거 참조 길이
NUM_NODES = 10         # N: 노드 개수
NUM_FEATURES = 1       # F: 피처 개수 (Bytes)
PRED_HORIZON = 1       # H: 예측할 미래 스텝 수

# --- Data Split & Seed ---
TRAIN_RATIO = 0.8      # 학습 데이터 비율 (80%)
RANDOM_SEED = 42       # 재현성을 위한 시드 고정
TOTAL_SAMPLES = 5000   # 생성할 전체 시뮬레이션 샘플 수 (약 7시간 분량)
