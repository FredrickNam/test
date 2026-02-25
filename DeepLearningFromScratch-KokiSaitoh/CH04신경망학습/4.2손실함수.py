# 4.2 손실함수
"""
손실함수 개념
- 신경망 성능의 나쁨을 나타내는 지표임
- 이 지표를 낮추는 것이 머신러닝의 과제임 (음수 처리하여 성능이 얼마나 좋은지에 대한 지표로도 사용 가능)

오차제곱합 (Sum Squared Error, SSE)
- 오차를 제곱하여 합산한 것
- 수식:
  $$ E = \frac{1}{2}\sum_{k}(y_k-t_k)^2 $$
  $$ y_k = 신경망의 추정 값$$
  $$ t_k = 정답레이블 $$
  $$k = 데이터의 차원 수(쉽게 말해 출력층의 노드 수)$$

교차엔트로피 오차 (Cross Entropy Error, CEE)
- 주로 분류 문제에 사용됨
- 수식:
  $$ E = -\sum_{k}t_klog(y_k) $$
  $$ t_k = \text{정답레이블} $$
  $$ y_k = \text{모델예측값} $$
  $$ k = \text{출력층의 노드 수} $$

미니배치 학습과 CEE 적용
- 미니배치: 훈련 데이터에 대한 손실함수를 모두 구하기 힘들 때, 데이터의 일부를 추려 전체의 근사치로 활용하는 것 (Batch 개념)
- 수식:
  $$E = - \frac{1}{N} \sum_{n} \sum_{k} t_{nk} \log y_{nk}$$
- $N$ = 배치 크기
- $t_{nk}$ = $n$번째 데이터의 정답 레이블
"""
import sys, os
import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def sum_squared_error(y,t):
    return 0.5

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize = True, one_hot_label=True)
x_train.shape

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size