# 김호중 딥러닝 기반 무선 네트워크 최적화 기법 개발

# LSTM 기반 트래픽 예측 및 임계값 자원 할당

## (1) 프로젝트 개요

본 프로젝트는 LSTM(Long Short-Term Memory) 기반 시계열 예측 모델을 활용하여  
네트워크 트래픽을 사전에 예측하고, 예측 결과를 기반으로 자원을 동적으로
할당하는 시스템을 구현하는 것을 목표로 한다.

기존의 현재 트래픽 수치 기반 자원 할당 방식은
급격한 트래픽 증가에 대한 대응이 지연될 수 있다.
이에 본 프로젝트에서는 예측 기반 자원 할당을 통해
서비스 품질을 유지하면서 자원 사용 효율을 향상시키고자 한다.

---

## (2) 시스템 아키텍쳐

시스템은 다음과 같은 단계로 구성된다.

1. 트래픽 로그 데이터 수집 및 전처리
   
2. 전처리된 시계열 데이터를 LSTM 모델에 입력
   
3. LSTM 모델을 통한 미래 트래픽 예측
   
4. 예측 결과를 기반으로 임계값 판단 및 자원 할당 수행

이를 통해 트래픽 변화에 선제적으로 대응 가능한
예측 기반 자원 관리 구조를 설계하였다.

---

## (3) 상세 내용

### 1) LSTM 기반 트래픽 예측 모델

트래픽 데이터의 시간적 의존성과 반복 패턴(계절성),
그리고 장기적인 변화 추세를 학습하기 위해
다층(Stacked) LSTM 구조를 사용한다.

- **Input Shape**: `(Batch_size, Time_steps, Features)`
- **Optimizer**: Adam Optimizer
- **Loss Function**: Mean Squared Error (MSE)

모델은 과거 일정 기간의 트래픽 시계열 데이터를 입력으로 받아
미래 특정 시점의 트래픽 부하를 예측하는 회귀 모델로 구성된다.

### 2) 예측 기반 임계값 자원 할당 로직

자원 할당은 단순한 현재 트래픽 수치가 아닌,
LSTM 모델이 예측한 미래 트래픽 값을 기준으로 수행한다.

- **Upper Threshold**
  - 예측 트래픽 > 전체 처리 용량의 80%
  - 자원 확장 (Scale-out)

- **Lower Threshold**
  - 예측 트래픽 < 전체 처리 용량의 20%
  - 자원 회수 (Scale-in)

임계값은 사전 실험을 통해 설정되며,
불필요한 잦은 Scale-in / Scale-out을 방지하기 위해
연속적인 조건 만족 시에만 자원 조정을 수행한다.

### 3) Example Code Snippet

본 프로젝트의 핵심 동작 흐름을 이해하기 위해
LSTM 입력 형태와 예측 기반 자원 할당 로직의 예제 코드를 제시한다.
아래 코드는 전체 구현이 아닌 주요 개념을 설명하기 위한 예시 코드이다.

1. LSTM 입력 데이터 형태 예시
import torch

(batch_size, time_steps, features)
예: 32개의 샘플, 과거 24 타임스텝, 1개의 트래픽 특성

input_sequence = torch.randn(32, 24, 1)

LSTM 모델 예측
predicted_traffic = model(input_sequence)

#해당 입력 구조를 통해 모델은 과거 일정 기간의 트래픽 시계열을 기반으로
#미래 트래픽 부하를 예측한다.

2. 예측 기반 임계값 자원 할당 로직 예시
def allocate_resource(predicted_load, total_capacity):
    if predicted_load > total_capacity * 0.8:
        scale_out()   # 자원 확장
    elif predicted_load < total_capacity * 0.2:
        scale_in()    # 자원 회수


#LSTM 모델의 예측 결과가 사전에 정의된 임계값을 초과하거나 미만일 경우,
#자원을 동적으로 확장(Scale-out) 또는 회수(Scale-in)한다.

3. 전체 실행 흐름 예시
predicted_traffic = model(input_sequence)
allocate_resource(predicted_traffic, total_capacity)


#위 흐름을 통해 트래픽 증가를 사전에 감지하고
#서비스 품질 저하 없이 자원을 선제적으로 관리할 수 있다.

---

## (4) 성능 평가 지표

모델의 예측 정확도와 시스템의 효율성은 다음 지표를 통해 종합적으로 평가한다.

| 분류 | 지표 명칭 | 설명 |
|------|----------|------|
| 모델 성능 | RMSE | 예측값과 실제값 간 차이의 평균 제곱근 오차 |
| 모델 성능 | MAE | 예측값과 실제값 간 절대 오차의 평균 |
| 시스템 효율 | Resource Utilization | 할당된 자원 대비 실제 사용된 자원의 비율 |
| 안정성 | SLA Violation Rate | 트래픽 폭주로 인해 서비스 수준 협약(SLA)을 위반한 비율 |

단일 예측 정확도 지표에 의존하지 않고,
모델 성능, 자원 효율성, 서비스 안정성을 함께 평가함으로써
예측 기반 자원 할당 시스템의 실질적인 효과를 분석한다.

---

## (5) 파일 구조

├── data/
│ ├── raw/ # 원본 트래픽 데이터
│ └── processed/ # 전처리된 데이터
│
├── model/
│ ├── lstm_model.py # LSTM 모델 정의
│ └── train.py # 모델 학습 코드
│
├── resource/
│ └── allocator.py # 임계값 기반 자원 할당 로직
│
├── evaluation/
│ └── metrics.py # 성능 평가 지표 계산
│
├── main.py # 전체 시스템 실행 파일
└── README.md

## 실행 환경

-OS: Linux (Ubuntu 20.04+) / Windows 10 & 11

-Language: Python 3.8 이상

-Key Libraries: PyTorch, Pandas, NumPy, Scikit-learn

-의존성 설치: ```bash pip install -r requirements.txt

## 실행 방법

-관련 라이브러리 설치: pip install -r requirements.txt

-모델 학습: python src/model.py --train

-시스템 실행: python main.py
- NumPy, Pandas
