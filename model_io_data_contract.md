# Model I/O Data Contract (공통 입력·출력 동일화 기획안)

> 목적: **용상(현재 기반 할당 모델)** 과 **호중(LSTM 예측 기반 모델)** 이 **동일한 입력 데이터/동일한 출력 규격**을 사용하도록 “데이터 계약(Data Contract)”을 고정한다.  
> 결과: 예나가 이 계약만 보고 **공통 트래픽(합성/실측) + 전처리 파이프라인**을 설계/구현할 수 있으며, 두 모델은 **공정 비교**가 가능하다.

---

## 1. 공통 개념 정의

### 1.1 노드(Node)
- `node_id`: 트래픽을 관측/할당할 단위(사이트, 셀, 서비스, AP 등)
- `N = num_nodes`: 노드 개수
- 노드 순서 고정: `node_0, node_1, ... node_{N-1}`  
  (훈련/추론/평가에서 항상 동일한 인덱스 순서를 유지)

### 1.2 시간(Time)
- `interval_sec`: 트래픽 샘플링 간격(초)
- `T = window_size`: 모델 입력에 포함되는 과거 타임스텝 길이
- `H = horizon`: 예측 기반 모델에서 예측할 미래 스텝(기본 1)

---

## 2. 공통 원본 로그 포맷 (Collector Output)

Collector는 실측/합성 여부와 무관하게 아래 컬럼을 **반드시** 출력한다.
해당 필드는 트래픽의 출력 스키마를 의미한다.

### 2.1 CSV/DB 레코드 스키마
| 필드 | 타입 | 설명 |
|---|---|---|
| `timestamp` | datetime | 수집 시각(UTC 또는 로컬 고정) |
| `node_id` | str | 노드 식별자(예: `node_3`) |
| `bytes` | int | 해당 interval 동안 전송된 바이트 수 |
| `packets` | int (옵션) | 해당 interval 동안 전송된 패킷 수 |

- 최소 필수: `timestamp, node_id, bytes`
- `packets`는 있으면 확장 feature로 사용 가능(없어도 동작해야 함)

---

## 3. 공통 전처리 산출물 (Preprocessing Output)

전처리는 원본 로그를 모델 입력 텐서로 변환한다.
텐서로 변환하는 이유는 딥러닝 모델의 학습을 위해서 임

### 3.1 입력 텐서 X (공통)
입력 텐서 작성 코드 방식은 추후 작성 예정
**Shape**
- `X.shape = (B, N, T, F)`

| 축 | 의미 |
|---|---|
| `B` | 배치 크기 |
| `N` | 노드 개수 |
| `T` | 과거 윈도우 길이 |
| `F` | feature 수 |

**기본 feature 구성(권장)**
- `F = 1`, `features = ["bytes"]`

> 참고: 호중님 LSTM 문서의 `(B, Time_steps, Features)`는 **노드별 시퀀스**를 의미한다.  
> 즉, 시스템 전체 관점에서는 `(B, N, T, F)`가 맞고, 모델 내부에서 노드별로 `(B, T, F)`를 사용한다.

### 3.2 타깃 텐서 (선택: 학습 목적에 따라)
#### (A) 자원 비율 타깃 Y_alloc (지도학습 할당 모델용)
- `Y_alloc.shape = (B, N)`
- 의미: 노드별 자원 **비율** (합=1) 용상 모델의 타깃 텐서

#### (B) 미래 트래픽 타깃 Y_pred (예측 모델용)
- `Y_pred.shape = (B, N, H, F)` 또는 간단히 `Y_pred.shape = (B, N)` (H=1, F=1이면 축 생략 가능)
- 의미: 미래 `H`스텝의 트래픽 예측 타깃 호중 모델의 타깃 텐

---

## 4. 공통 모델 출력 규격

### 4.1 공통 출력: 자원 비율 P
모든 모델은 최종적으로 노드별 자원 비율을 반환한다.

- `P.shape = (B, N)`
- 제약: 각 샘플에 대해 `sum(P[b, :]) = 1`
- 권장 구현: `softmax(logits, dim=1)`

### 4.2 자원량 변환(환경 파라미터)
총 자원량 `C`가 있을 때,
- `alloc = P * C`
- `alloc.shape = (B, N)`

---

## 5. 모델별 처리 흐름 (입출력은 동일, 내부만 다름)

### 5.1 용상(현재 기반 동적 할당; MLP 등)
- 입력: `X (B, N, T, F)`
- 처리: 최근 패턴을 기반으로 즉각 비율 산출
- 출력: `P (B, N)`

**Flow**
1) `X`를 적절히 flatten/embedding  
2) `logits = model(X)`  
3) `P = softmax(logits, dim=1)`  
4) `alloc = P * C`

### 5.2 호중(LSTM 예측 기반 할당; Stacked LSTM)
- 입력: `X (B, N, T, F)`
- 처리: 노드별 시계열을 LSTM으로 예측 → 예측값을 기준으로 비율 산출(또는 임계값 로직)
- 출력: `P (B, N)`  (**동일 규격 유지**)

**권장 Flow (공정 비교/단일 출력 유지 버전)**
1) 노드별로 `X[:, i, :, :]` → `(B, T, F)`  
2) `pred_i = LSTM(X_i)` → 미래 트래픽(예: `(B, 1)` 또는 `(B, H, F)`)  
3) `pred = stack(pred_i)` → `(B, N)`  
4) `logits = g(pred)` 또는 `P = softmax(pred, dim=1)` (예측 트래픽 기반 비율)  
5) `alloc = P * C`

**임계값(scale in/out) 로직은 “비교 실험용 정책”으로 별도 관리**
- 예측값이 `> 0.8*C`이면 `C` 자체를 늘리는 정책(Scale-out)  
- 예측값이 `< 0.2*C`이면 `C`를 줄이는 정책(Scale-in)  
- 단, 본 계약에서는 **모델 출력은 항상 `P (B,N)`로 고정**하고, Scale 정책은 상위 컨트롤러에서 수행한다.

---

## 6. 공통 파라미터(팀 합의 필요 항목)

아래 값은 **한 번 정하면 팀 전체가 동일하게 사용**한다.

| 파라미터 | 예시 | 설명 |
|---|---:|---|
| `interval_sec` | 5 | 샘플링 간격(초) |
| `window_size (T)` | 12 또는 24 | 과거 윈도우 길이 |
| `features (F)` | 1 | bytes만 쓰면 1 |
| `num_nodes (N)` | 10 | 노드 개수 |
| `capacity (C)` | 100 | 총 자원량(단위는 프로젝트 정의) |
| `horizon (H)` | 1 | 예측 스텝(호중 모델용) |

---

## 7. 공통 테스트(수용 기준)

### 7.1 데이터 파이프라인 수용 기준
- Collector가 생성한 raw 로그로부터 전처리 결과 `X`가 반드시 `(B, N, T, F)`를 만족
- node 인덱스가 매 실행마다 바뀌지 않음
- 결측 timestamp가 존재할 경우 정책(0 채움/보간/샘플 드랍)을 문서화

### 7.2 모델 I/O 수용 기준
- 두 모델 모두 동일 입력 `X`를 받아 **항상 동일 shape `P (B,N)` 출력**
- `P`는 각 샘플에 대해 합이 1(softmax)
- `alloc = P*C`가 음수 없이 정상 산출

---

## 8. 권장 파일 구조(공통 계약 반영)

```
traffic-resource-allocation/
├─ collector/
│  └─ INFO.md
├─ preprocessing/
│  └─ INFO.md
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
│  ├─ allocator_mlp.py        # 용상
│  └─ allocator_lstm.py       # 호중
├─ train/
│  ├─ train_allocator.py
│  └─ train_predictor.py      # (선택)
└─ evaluation/
   └─ metrics.py
```

---

## 9. 한 줄 요약(발표/보고서용)

> “트래픽 수집/전처리 파이프라인을 공통 모듈로 고정하고, 입력 텐서 `(B, N, T, F)` 및 출력 자원 비율 `(B, N)`을 동일화하여 서로 다른 딥러닝 기법(현재 기반 vs 예측 기반)을 공정하게 비교한다.”
