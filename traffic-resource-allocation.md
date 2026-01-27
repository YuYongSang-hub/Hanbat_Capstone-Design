# Traffic Resource Allocation using LSTM
딥러닝 기반 트래픽 예측을 활용하여 네트워크 자원을 노드별로 동적으로 할당하는 실험 프로젝트입니다.

본 프로젝트는 LSTM 기반 예측 모델과 임계값(Threshold) 정책을 결합하여 선제적(Proactive) 자원 분배의 효용성을 검증합니다.

📌 Capstone Design 프로젝트

📌 Model-agnostic 데이터 파이프라인 + LSTM baseline

---

## 📌 프로젝트 개요

- **문제 정의**: 고정된 자원 할당으로 인한 트래픽 폭주 시의 병목(Bottleneck) 및 저부하 시의 자원 낭비 문제 해결.
- **해결 방안**: 과거 트래픽 패턴을 학습하여 미래 수요를 예측하고, 시스템 사용률에 따라 자원을 동적으로 확장(Scale-out)하거나 회수(Scale-in).

---

## 🧱 시스템 아키텍처 및 파이프라인

```
[Generator] ----> Raw Traffic CSV ----> [Preprocessor] ----> (B, N, T, F) Tensor
                                                                    |
[Decision] <---- [Threshold Policy] <---- [LSTM Model] <-----------+
(Scale Out/In)

```
---

## 📊 데이터 규격 및 입력 형태

🔹 **입력 텐서 (Model Input: X)**
- **Shape**: (B, N, T, F)
- **의미**:
  
  B: Batch size (학습 단위)
  
  N: 노드 수 (가용 자원 단위)
  
  T: 과거 시간 윈도우 길이 (참조 데이터)
  
  F: 트래픽 피처 수 (기본: Bytes)

  🔹 **타깃 데이터 (Target: Y)**

  - **Shape**: (B, N)

  - **의미**: 각 노드별 시점 t+1에서의 트래픽 발생량 (Regression)

  ---

 ## 🧠 LSTM 기반 예측 모델

 - **Parameter Sharing**: 모든 노드가 동일한 LSTM 가중치를 공유하여 학습 효율 극대화
 - **Baseline**: 복잡한 구조 대신 시계열 특성 추출에 충실한 2-Layer LSTM 구조 채택
 - **Output**: 로그 스케일링된 예측 트래픽 값

---

 ## 🚀 실행 방법

 **환경 구축**
 ```
 pip install -r requirements.txt
 ```
 **전체 파이프라인 실행**

 데이터 생성부터 전처리, 학습, 추론까지 한 번에 수행합니다.
 ```
 python main.py
 ```
**개별 모듈 실행**

데이터 생성: `python generator.py`

전처리: `python preprocessor.py`

모델 학습: `python train.py`

결과 추론: `python inference.py`

---

## 📈 실험 지표 (Metrics)

|지표| 설명 | 비고|
|------|----------|------|
|MSE | 실제 트래픽과 예측 트래픽 간의 오차|모델 성능 지표|
|Scaling Count|Scale-out/in 발생 횟수|정책 작동 빈도|
|Utilization|할당된 자원 대비 실제 트래픽 비율|자원 효율성 지표|

## 모델 및 실험 PARAMETERS

- Data Generation

| Parameter | Value | Description |
|--------|------|------------|
| INTERVAL_SEC | 5 sec | Traffic sampling interval |
| TOTAL_SAMPLES | 5000 | Total time steps (~7 hours) |
| NUM_NODES | 10 | Number of network nodes |

- Input Window

| Parameter | Value | Description |
|--------|------|------------|
| WINDOW_SIZE (T) | 12 | Past time steps used as input |
| NUM_FEATURES (F) | 1 | Bytes per node |

- Model (LSTM)

| Parameter | Value |
|--------|------|
| Hidden Size | 64 |
| Num Layers | 2 |
| Optimizer | Adam |
| Loss | MSE |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Batch Size | 32 |

- Inference & Allocation

| Parameter | Value | Description |
|--------|------|------------|
| TOTAL_CAPACITY | 100 | Total available resource |
| SCALE_OUT_TH | 0.8 | Scale-out threshold |
| SCALE_IN_TH | 0.2 | Scale-in threshold |
