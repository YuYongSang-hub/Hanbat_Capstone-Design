import numpy as np

# npz 파일 경로
path = "data/processed/traffic_data_train.npz"

data = np.load(path)

print("📦 npz 안에 들어있는 키들:")
print(data.files)

x = data["x_data"]
y = data["y_data"]

print("\n📐 데이터 shape")
print("x_data:", x.shape)
print("y_data:", y.shape)

print("\n🔍 첫 번째 샘플 요약")
print("x_data[0] shape:", x[0].shape)
print("y_data[0] shape:", y[0].shape)

print("\n📊 예시 값")
print("node_0 과거 12스텝:", x[0, 0, :, 0])
print("node_0 다음 스텝:", y[0, 0])
