# 6.5 적절한 하이퍼파라미터 찾기
"""
데이터셋의 3분할
- 하이퍼파라미터(뉴런 수, 배치 크기, 학습률, 가중치 감소율 등)를 평가할 때 시험 데이터를 사용하면 모델이 시험 데이터 셋에 간접적으로 과대적합됨
- 이를 방지하기 위해 전체 데이터셋을 훈련 데이터, 검증 데이터, 시험 데이터 3가지로 분리 사용함
- 훈련 데이터: 가중치 및 편향 매개변수 실질적 학습에 사용함
- 검증 데이터: 여러 하이퍼파라미터 조건의 성능 평가 잣대로 사용함
- 시험 데이터: 모델 완성 후 최종 범용 성능 측정에만 사용함

하이퍼파라미터 최적화 단계
- 일정 간격의 그리드 탐색(Grid Search)보다 무작위 탐색(Random Search)이 더 우수한 결과를 도출한다고 알려짐
- Step 0: 하이퍼파라미터 값의 대략적인 범위를 넓게 설정함
- Step 1: 설정된 범위 안에서 하이퍼파라미터 값을 무작위로 추출함
- Step 2: 샘플링한 값으로 학습 진행 후 검증 데이터로 정확도를 평가함 (탐색 속도를 위해 에포크 수를 아주 작게 단축 설정함)
- Step 3: Step 1과 2 과정을 일정 횟수 반복하며 얻어진 결과 동향을 바탕으로 최적값의 범위를 점진적으로 좁혀나감
"""
import sys, os
import numpy as np

# 부모 디렉터리의 모듈 참조 경로 추가 및 필요 함수 임포트함
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.util import shuffle_dataset

# 데이터 로드함
(x_train, t_train), (x_test, t_test) = load_mnist()

# 훈련 데이터를 검증용으로 분리하기 전 특정 레이블 편향 방지를 위해 원본 데이터를 뒤섞음
x_train, t_train = shuffle_dataset(x_train, t_train)

# 검증 데이터의 할당 비율을 전체의 20%로 설정함
validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)

# 뒤섞인 전체 훈련 데이터에서 앞부분을 검증용 데이터 변수로 분리함
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

# 분리하고 남은 나머지 영역 데이터를 실제 훈련 데이터 변수로 덮어씌워 갱신함
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]