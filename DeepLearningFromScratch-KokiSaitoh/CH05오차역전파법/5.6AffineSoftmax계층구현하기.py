# 5.6 Affine/Softmax 계층 구현하기
"""
Affine 계층
- 신경망의 순전파 시 가중치 신호의 총합을 구하기 위해 행렬 내적 연산(np.dot)을 수행하는 계층임
- 스칼라가 아닌 행렬이 흐르며, 역전파 시 전치 행렬을 사용하여 형상(Shape)을 일치시키는 것이 핵심임
- 수식: $$ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}*W^T $$

Softmax-with-loss 계층
- 입력된 점수를 확률로 변환하는 Softmax 계층과, 손실을 구하는 교차엔트로피 오차(CEE) 계층을 결합함
- 신경망 학습의 목적은 예측과 정답의 오차를 줄이는 것임
- Softmax와 CEE를 결합하여 역전파를 구하면, 오차인 $(y_k - t_k)$가 앞 계층으로 깔끔하게 전달되어 학습 효율이 상승함
"""
import numpy as np

# 사전 의존성 함수: 교차 엔트로피 오차 및 소프트맥스 함수 정의
def softmax(x):
    # 배치 처리를 고려한 소프트맥스 연산 수행함
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    # 정답 레이블과 예측값을 비교하여 교차 엔트로피 손실을 계산함
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Affine:
    def __init__(self, W, b):
        # 가중치와 편향, 그리고 해당 미분값을 저장할 인스턴스 변수 초기화
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 입력 데이터 저장 후 행렬 내적 연산과 편향 덧셈을 수행함
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        # 가중치의 전치 행렬을 내적하여 입력 데이터 방향의 기울기 산출함
        dx = np.dot(dout, self.W.T)
        # 입력 데이터의 전치 행렬을 내적하여 가중치 방향의 기울기 산출함
        self.dW = np.dot(self.x.T, dout)
        # 배치 방향(axis=0)으로 합산하여 편향의 기울기 산출함
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        # 손실, 예측 결과, 정답 레이블을 보관할 변수
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        # 정답 레이블을 저장하고, 소프트맥스 후 손실을 계산함
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        # 배치 크기만큼 나누어 데이터 1개당 오차를 역전파로 앞 계층에 전달함
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx