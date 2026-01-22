# config.py

# --- Path Settings ---
RAW_DATA_PATH = "data/raw/traffic_log.csv"
PROCESSED_DATA_DIR = "data/processed"

# --- Traffic & Data Spec ---
INTERVAL_SEC = 5
WINDOW_SIZE = 12        # T
NUM_NODES = 10          # N
NUM_FEATURES = 1        # F
PRED_HORIZON = 1

# --- Dataset ---
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
TOTAL_SAMPLES = 5000

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# --- Inference ---
TOTAL_CAPACITY = 100
SCALE_OUT_TH = 0.8
SCALE_IN_TH = 0.2
