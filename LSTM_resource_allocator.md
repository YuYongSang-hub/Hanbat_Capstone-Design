# LSTM 기반 트래픽 예측 → 자원 비율 출력 모델
# 1️⃣ 설계 개요 (계약 기반)

입력: `X ∈ ℝ^{B×N×T×F}`

처리 방식:

- 노드별 시계열 `(T, F)` 를 공유 LSTM으로 처리

- 각 노드의 미래 트래픽 스칼라 예측

출력:

- 노드별 자원 비율 `P ∈ ℝ^{B×N}`

- `sum(P[b, :]) = 1`

👉 Data Contract 완전 준수

---

# 2️⃣ 모델 구조 설명
```
X (B, N, T, F)
 └─ reshape → (B·N, T, F)
     └─ Stacked LSTM
         └─ Last Hidden State
             └─ Linear → traffic score
                 └─ reshape → (B, N)
                     └─ Softmax(dim=1) → P
```

---

# 3️⃣ PyTorch LSTM 모델 구현

```
import torch
import torch.nn as nn

class LSTMResourceAllocator(nn.Module):
    def __init__(
        self,
        input_size=1,      # F
        hidden_size=64,
        num_layers=2,
        num_nodes=10       # N
    ):
        super().__init__()

        self.num_nodes = num_nodes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 노드별 예측 트래픽 → 스칼라
        self.fc = nn.Linear(hidden_size, 1)

        # 자원 비율 보장
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        """
        X: (B, N, T, F)
        return: P (B, N)
        """
        B, N, T, F = X.shape
        assert N == self.num_nodes, "Node count mismatch"

        # (B, N, T, F) → (B*N, T, F)
        X = X.view(B * N, T, F)

        # LSTM
        _, (h_n, _) = self.lstm(X)

        # 마지막 레이어의 hidden state
        h_last = h_n[-1]              # (B*N, hidden_size)

        # 트래픽 예측 스칼라
        pred = self.fc(h_last)        # (B*N, 1)

        # (B*N, 1) → (B, N)
        pred = pred.view(B, N)

        # 자원 비율 출력
        P = self.softmax(pred)        # (B, N), sum=1

        return P

```

---

# 4️⃣ 출력 검증 코드 (Data Contract 검증)

모든 노드 자원 비율 합 = 1 인지 확인
```
def validate_output(P, tol=1e-6):
    """
    P: (B, N)
    """
    sums = P.sum(dim=1)
    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=tol
    ), "❌ Resource allocation does not sum to 1"

    print("✅ Output validation passed: sum(P) == 1")
```

- 사용 예시
```
B, N, T, F = 4, 10, 12, 1
X = torch.randn(B, N, T, F)

model = LSTMResourceAllocator(
    input_size=F,
    hidden_size=64,
    num_layers=2,
    num_nodes=N
)

P = model(X)
validate_output(P)
```
---

# 5️⃣ 설계 장점 
- ✅ Data Contract 준수

입력 (B, N, T, F)

출력 (B, N)

Softmax로 합 = 1 보장

- ✅ 공정 비교 가능

MLP / CNN 기반 모델과 출력 규격 동일

동일한 alloc = P * C 적용 가능

- ✅ 확장성

F > 1 (packets, latency 등) 확장 가능

horizon > 1 예측으로 쉽게 확장 가능

---

6️⃣ README / 보고서용 한 줄 설명

“노드별 시계열 트래픽을 공유 LSTM으로 예측한 후, Softmax를 통해 자원 비율을 산출하는 예측 기반 자원 할당 모델이다.”
