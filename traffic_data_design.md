# 트래픽 데이터 정의 및 전처리 파이프라인

## 개요

공정한 성능 비교 위해 데이터 파이프라인(수집/전처리)과 트래픽 설계 원칙 정의.


**동일한 입력 데이터**를 전제로, 트래픽 해석에 따른 서로 다른 접근 방식의 차이 비교를 위해 모델에 종속되지 않는 중립적이고 재현 가능한 실험 환경을 구축한다.


가이드라인에 따라 모든 모델은 동일한 입력 데이터 (x)와 동일한 출력 규격(p)를 공유한다.

*트래픽 설계 원칙 및 특성은 공통 파라미터 설정에 대한 근거를 명시적으로 제시하기 위해 추가한 항목이므로 실제 알고리즘 설계하실 때 공통 파라미터 항목 위주로 확인 부탁드립니다.*

---

## 합의 사항

모든 모델은 동일한 입력 데이터(`X`)와 동일한 출력 규격(`P`)을 공유한다.

### 1.1 핵심 데이터 흐름
1.  **Collector:** 시뮬레이터 또는 로그 생성기에서 원본 데이터(Raw Data) 수집
2.  **Preprocessing:** 시계열 데이터를 모델 학습용 텐서 형태로 변환
3.  **Model Input:** 두 모델 모두 `(B, N, T, F)` 형태의 동일한 텐서를 입력받음

### 1.2 공통 파라미터 (팀 전체 고정)
다음 파라미터는 실험의 일관성을 위해 고정하며, 변경 시 팀 합의가 필요하다.

| 변수명 | 값(예시) | 설명 |
|---|:---:|---|
| `interval_sec` | **5** | 트래픽 샘플링 간격 (초) |
| `window_size` (**T**) | **12** | 모델이 참조할 과거 시간 단계 |
| `features` (**F**) | **1** | 입력 피처 수 (기본: `bytes` 1개) |
| `num_nodes` (**N**) | **10** | 자원을 할당받는 노드(사용자) 수 |
| `batch_size` (**B**) | **32** | 학습 배치 크기 (가변 가능) |

---

## 2. 트래픽 설계 원칙 및 특성

제공되는 트래픽 데이터는 단순한 난수 생성이 아닌, 비교 실험의 유의미성을 확보하기 위해 다음 원칙에 따라 설계된다.

### 2.1 설계 원칙
1.  **모델 중립성:** 데이터는 특정 모델(CNN, LSTM 등)에 유리한 패턴으로 편향되지 않으며, 모든 모델은 동일한 입력 텐서를 공유한다.
2.  **재현 가능성:** 동일한 시드(Seed)와 파라미터 설정을 통해 언제든 동일한 트래픽 패턴을 재생성할 수 있어야 한다.
3.  **예측 효용성 검증:** 현재 정보만으로 충분한 구간과, 미래 예측이 성능 향상에 기여할 수 있는 구간을 혼합하여 설계한다.

### 2.2 트래픽 핵심 특성
실제 네트워크 환경을 모사하기 위해 다음과 같은 통계적 특성을 반영한다.

* **시간 변동성:** 완만한 변화 구간과 급격한 트래픽 폭주(Burst) 구간을 모두 포함한다.
* **노드 간 불균형:** 모든 노드가 균등한 트래픽을 갖지 않는다. 특정 노드에 트래픽이 집중되는 상황을 포함하여 자원 할당의 형평성을 테스트한다.
* **패턴과 추세:** 단기적인 노이즈와 중장기적인 증감 추세를 혼합하여, 모델이 유의미한 패턴을 학습할 수 있도록 한다.

### 2.3 데이터셋 분할 및 셔플 정책
* Chronological Split: 시계열 데이터의 특성상, 학습(Train)과 테스트(Test) 데이터는 시간 순서를 기준으로 분할한다.
    예: 전체 데이터의 앞쪽 80%를 학습용, 뒤쪽 20%를 테스트용으로 할당.
* No Shuffling: 데이터 생성 및 배치 구성 시, 시간적 연속성을 파괴하는 Random Shuffle은 금지된다. (단, 학습 시 배치 내부에서의 순서는 모델 특성에 따라 달라질 수 있음)

---

## 3. 데이터 스키마 정의

### 3.1 원본 로그 규격
Collector가 생성하는 CSV 또는 로그 데이터의 필수 컬럼.

| 컬럼명 | 타입 | 필수 여부 | 설명 |
|---|---|:---:|---|
| `timestamp` | datetime | ✅ | 수집 시각 (UTC 권장) |
| `node_id` | string | ✅ | 노드 식별자 (예: `node_0`, `node_1` ...) |
| `bytes` | integer | ✅ | 해당 Interval 동안 발생한 트래픽 양 |
| `packets` | integer | (옵션) | 해당 Interval 동안 발생한 패킷 수 |

> **Note:** `node_id`는 항상 `node_0`부터 `node_{N-1}`까지 고정된 순서로 정렬되어 처리된다.

### 3.2 전처리 후 입력 텐서 규격 (`X`)
모든 모델이 공통으로 사용할 입력 데이터의 형태이다.

* **변수명:** `X`
* **Shape:** `(B, N, T, F)`
* **Data Type:** `float32`

| 차원 | 크기 | 의미 |
|:---:|:---:|---|
| **B** | 32 | 배치 크기 |
| **N** | 10 | 노드(사용자) 개수 |
| **T** | 12 | 과거 윈도우 길이 |
| **F** | 1 | 트래픽 특성 수 (예: `[bytes]`) |

#### 데이터 해석 예시
`X[0, 3, :, 0]` : 첫 번째 배치 샘플에서, **3번 노드**의 **과거 T시간** 동안의 **트래픽 발생량** 시계열 데이터.

### 3.3 타깃 데이터 규격 (`Y`)
모델의 학습 목적에 따라 Y값은 달라지지만, 전처리 파이프라인에서는 이를 지원할 수 있는 형태로 데이터를 제공한다.

1.  **할당 모델용 타깃 (`Y_alloc`):** `(B, N)` - 시점 $t$에서의 최적 자원 분배 비율
2.  **예측 모델용 타깃 (`Y_pred`):** `(B, N)` - 시점 $t+1$에서의 트래픽 발생량

---

## 4. 전처리 파이프라인 로직

아래 로직을 통해 원본 로그를 입력 텐서 `X`로 변환한다.

### Step 1. Pivot & Cleaning
* 원본 로그를 `(시간 x 노드)` 형태의 매트릭스로 변환한다.
* 누락된 `timestamp`나 `node_id`는 `0`으로 채운다.
* 노드 컬럼의 순서는 `node_id` 오름차순으로 고정한다.

### Step 2. Sliding Window
* 시계열 데이터를 `window_size` 길이만큼 잘라 샘플을 생성한다.
* 예: 데이터 길이가 100이고 `T=10`이면, 91개의 샘플이 생성된다.

### Step 3. Dimension Expansion
* 모델 입력 규격 `(B, N, T, F)`에 맞게 차원을 확장한다.
* `(Samples, T, N)` $\rightarrow$ `(Samples, N, T)` $\rightarrow$ `(Samples, N, T, F)`

### Step 4. Data Scaling
* 원본 bytes 값은 스케일링 없이 딥러닝 모델에 입력 시 학습 불안정 유발 가능성이 있음.
* 전처리 단계에서 Log Scaling (np.log1p) 또는 Min-Max Scaling을 적용하여 값을 0 ~ 1 또는 작은 범위로 변환하여 제공할 예정이므로, 모델 입력 전 별도의 스케일링을 수행할 필요는 없다.

### (참고) 전처리 핵심 코드 예시
```python
import numpy as np
import pandas as pd

def create_dataset(df, window_size):
    # 1. Pivot: Index=Time, Columns=Nodes, Values=Bytes
    pivot = df.pivot(index="timestamp", columns="node_id", values="bytes").fillna(0)
    data = pivot.values  # Shape: (Time_steps, Num_nodes)
    
    # 2. Sliding Window
    X_list = []
    for i in range(len(data) - window_size):
        # (T, N) 형태로 자름
        window = data[i : i + window_size] 
        X_list.append(window)
        
    X = np.array(X_list)  # (Samples, T, N)
    
    # 3. Reshape for Model: (Samples, N, T) -> (Samples, N, T, F)
    X = np.transpose(X, (0, 2, 1)) # (Samples, N, T)
    X = X[..., np.newaxis]         # (Samples, N, T, 1)
    
    return X 
```

## 5. 모델 출력 규격

### 출력 변수
- **P** (자원 할당 비율)

### Shape
- `(B, N)`

### 제약 사항
- 각 노드 할당 비율 \(P\)는 `0 ≤ P ≤ 1` 범위에 있어야 함
- 각 샘플별 노드 할당 비율의 합은 1이어야 함  
  (`∑ P_i = 1`)
- 구현 시 **Softmax** 사용을 권장함


---

## 6. 폴더 구조 및 산출물

프로젝트의 데이터 및 전처리 모듈은 아래 경로에 위치한다.

```plaintext
traffic-resource-allocation/
├─ data/
│  ├─ raw/                  # 원본 CSV 로그
│  └─ processed/            # 전처리된 .npz 파일 (X, Y 텐서)
├─ preprocessing/
│  ├─ generator.py          # 트래픽 생성/수집기
│  ├─ preprocessor.py       # 전처리 로직
│  └─ config.py             # 공통 파라미터 설정
└─ ...
```

## 최종 산출물 포맷

전처리가 완료된 데이터는 **NumPy 포맷 (`.npz`)** 으로 저장하여 공유한다.

### 파일명 예시
traffic_data_train.npz

### Key 구성
- `x_data`: 입력 텐서 `(B, N, T, F)`
- `y_data`: 타깃 텐서 (필요 시)
- `timestamps`: 해당 샘플의 시각 정보 (디버깅용)

### 비고
각 모델 담당자는 위 명세서의 입력 텐서 규격에 맞춰 모델의 입력 레이어를 설계해 주시기 바랍니다.

## 6.1 소스 코드 상세 (Source Code Details)

아래 3개 파일을 같은 폴더에 두고 다음 명령어 순서대로 실행하시면 됩니다. 실행하시면 data/processed/ 폴더에 .npz 파일 추가됩니다.

```Bash
# 1. 가짜 트래픽 생성 (data/raw/traffic_log.csv 생성됨)
python generator.py
# 성공 시: Raw data generated at: data/raw/traffic_log.csv (폴더 안에 data/raw 폴더가 생기고 파일이 만들어집니다.)

# 2. 전처리 및 텐서 변환 (data/processed/traffic_data_train.npz 생성됨)
python preprocessor.py
# 성공 시: Data saved to: data/processed 식의 데이터 생성됐다는 말 나옵니다.
```

---

### `config.py` (설정 파일)

- 실험의 재현성과 일관성을 보장하기 위해 전역 파라미터  
  (예: `N`, `T`, `F`, `Random Seed`)를 통합 관리함
- 파라미터 변경 시 이 파일만 수정하면  
  **트래픽 생성기와 전처리기에 동시에 적용됨**

---

### `generator.py` (트래픽 생성기)

- 명세서의 **2. 트래픽 설계 원칙**을 반영하여 가상의 트래픽 로그를 생성함
- 다음 요소를 포함한 CSV 파일을 생성하여 `data/raw/` 경로에 저장함:
  - 시간적 패턴 (Sine wave)
  - 노이즈 (Noise)
  - 노드 간 트래픽 불균형
  - 돌발 트래픽 (Burst)

---

### `preprocessor.py` (전처리 파이프라인)

- Raw 데이터를 읽어와 모델 학습에 적합한 형태로 변환함
- 주요 기능:
  - 결측치 처리 (Zero-filling)
  - 로그 스케일링 (Log Scaling)
  - 슬라이딩 윈도우 (Sliding Window)
  - 차원 확장
  - 시계열 순서를 고려한 Train/Test 분할
- 실행 결과로 `data/processed/` 경로에 `.npz` 파일을 생성함


