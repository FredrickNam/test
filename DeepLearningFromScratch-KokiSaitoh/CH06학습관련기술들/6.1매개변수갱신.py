# 6.1 매개변수 갱신
"""
최적화(Optimization) 문제
- 신경망 학습의 목적은 손실함수의 값을 가능한 한 낮추는 최적의 매개변수를 찾는 것임
- 확률적 경사 하강법(SGD)은 기울기를 구해 기울어진 방향으로 매개변수를 갱신하는 단순한 방법임
- SGD의 단점: $f(x,y) = \frac{1}{20} x^2 + y^2$ 같은 비등방성(Anisotropy) 함수에서는 기울기가 최적점을 가리키지 않아 지그재그로 탐색하며 비효율적임

모멘텀(Momentum)
- 운동량을 뜻하며, 물리 법칙(가속도와 마찰)을 모방한 방법임
- 수식:
  $$ v \leftarrow \alpha v - \eta\frac{\partial L}{\partial W}$$  
  $$ W \leftarrow W + v $$
  (단, $\eta$: 학습률, $v$: 속도, $\alpha$: 마찰/공기저항 계수)
- x축 방향의 힘은 작지만 방향이 변하지 않아 한 방향으로 일정하게 가속되므로 지그재그 현상이 줄어듦

AdaGrad (학습률 감소 기법)
- 학습률이 너무 크면 발산하고 너무 작으면 학습 시간이 길어지므로, 처음엔 크게 학습하다 점차 줄이는 '학습률 감소(Learning Rate Decay)' 기법을 사용함
- AdaGrad는 개별 매개변수에 적응적으로(Adaptive) 학습률을 조정함
- 수식:
  $$\begin{aligned}
  h & \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W} \\
  W & \leftarrow W - \eta \frac{1}{\sqrt{h}} \odot \frac{\partial L}{\partial W}
  \end{aligned}$$
- $h$에 기울기 제곱을 누적하고, 갱신 시 $\frac{1}{\sqrt{h}}$를 곱해 많이 움직인 매개변수의 학습률을 낮춤

Adam
- 모멘텀(물리 법칙 모방)과 AdaGrad(적응적 학습률 조정)를 융합한 최적화 기법임
- 네 가지 방법 중 절대적으로 뛰어난 방법은 없으며 문제에 따라 적절히 선택해야 함
"""
import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        # 학습률(learning rate) 초기화함
        self.lr = lr

    def update(self, params, grads):
        # 각 매개변수에 대해 기울기 방향으로 학습률만큼 이동하여 값을 갱신함
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        # 학습률과 모멘텀(마찰 계수 역할) 초기화함
        self.lr = lr
        self.momentum = momentum
        # 속도(v) 변수를 딕셔너리로 사용하기 위해 초기화 보류함
        self.v = None

    def update(self, params, grads):
        # 최초 호출 시 매개변수와 동일한 형상의 0 배열로 속도(v) 딕셔너리를 생성함
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # 이전 속도에 마찰을 적용하고, 기울기에 따른 가속도를 더함
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # 계산된 속도를 매개변수에 더해 실제 위치를 갱신함
            params[key] += self.v[key]

class AdaGrid:
    def __init__(self, lr=0.01):
        # 학습률 및 기울기 제곱 누적합 변수 초기화함
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        # 최초 호출 시 매개변수 형상과 일치하는 0 배열로 누적합(h) 딕셔너리 생성함
        if self.h is None:
            self.h = {} 
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 기존 기울기의 제곱을 누적하여 더함
            self.h[key] += grads[key] * grads[key]
            # 누적합의 제곱근에 반비례하도록 학습률을 조정하여 매개변수 갱신함 (0으로 나누는 것 방지 위해 1e-7 더함)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)