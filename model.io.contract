1️⃣ LSTM 기반 모델 구조 (최종 출력: (B, N))

핵심 포인트

노드별 LSTM → 노드별 예측값 (B, N) 생성

마지막 레이어에 nn.Softmax(dim=1) 적용

출력은 자원 비율 P
```
import torch
import torch.nn as nn

class LSTMAllocator(nn.Module):
    def __init__(self, num_nodes, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes

        # 노드별 LSTM (공유 가중치)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 노드별 예측값을 scalar로 축소
        self.fc = nn.Linear(hidden_size, 1)

        # ⭐ 데이터 계약 준수: 노드 차원 기준 Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        """
        X: (B, N, T, F)
        return: P (B, N)
        """
        B, N, T, F = X.shape
        assert N == self.num_nodes

        node_preds = []

        for i in range(N):
            # 노드 i의 시계열 입력
            Xi = X[:, i, :, :]           # (B, T, F)
            lstm_out, _ = self.lstm(Xi)  # (B, T, H)
            last_hidden = lstm_out[:, -1, :]  # (B, H)

            pred_i = self.fc(last_hidden)     # (B, 1)
            node_preds.append(pred_i)

        # (B, N)
        preds = torch.cat(node_preds, dim=1)

        # ⭐ 최종 출력: 자원 비율
        P = self.softmax(preds)

        return P
```
2️⃣ 출력 검증 코드 (데이터 계약 준수 여부 확인)

아래 코드는 모든 배치에 대해 노드 합이 1인지 검증합니다.
```
def validate_allocation(P, tol=1e-6):
    """
    P: (B, N) - 모델 출력
    tol: 허용 오차
    """
    # shape 체크
    assert P.dim() == 2, f"Invalid shape: {P.shape}"

    # 음수 여부 확인
    if torch.any(P < 0):
        raise ValueError("❌ Allocation contains negative values")

    # 합이 1인지 확인
    row_sums = P.sum(dim=1)  # (B,)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tol):
        raise ValueError(
            f"❌ Allocation sum check failed: {row_sums}"
        )

    print("✅ Allocation validation passed (sum == 1 for all batches)")
```
3️⃣ 간단한 실행 예시
```
B, N, T, F = 4, 5, 12, 1

model = LSTMAllocator(num_nodes=N, input_size=F)
X = torch.rand(B, N, T, F)

P = model(X)  # (B, N)

print("P shape:", P.shape)
print("Row sums:", P.sum(dim=1))

validate_allocation(P)
```

출력 예:
```
P shape: torch.Size([4, 5])
Row sums: tensor([1., 1., 1., 1.])
✅ Allocation validation passed (sum == 1 for all batches)
```
4️⃣ 데이터 계약 관점에서의 의미 (중요)

✔ 입력: (B, N, T, F)

✔ 출력: (B, N)

✔ sum(P[b, :]) = 1 보장

✔ alloc = P * C 즉시 가능
