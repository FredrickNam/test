# 5.5 활성화 함수 계층 구현하기
"""
ReLU 계층 구현
- 입력이 0보다 크면 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 활성화 함수임
- 역전파 시, 순전파에서 0 이하였던 원소들의 미분값은 0으로 차단하여 하류로 전달함

Sigmoid 계층 구현
- 수식 $y = \frac{1}{1+e^{-x}}$을 계산하는 활성화 함수임
- 복잡한 수식이지만 노드를 나누어 연쇄법칙을 적용하면 역전파 결과를 깔끔하게 정리할 수 있음
- 결론적으로 Sigmoid 계층의 역전파는 순전파의 출력 y만으로 $\frac{\partial L}{\partial y} y(1-y)$ 식을 통해 간단히 계산 가능함
"""
import numpy as np

class Relu:
    def __init__(self):
        # 0 이하인 값의 위치를 판별할 불리언 마스크 변수 선언
        self.mask = None

    def forward(self, x):
        # x 배열에서 0 이하인 원소의 인덱스를 True로 마스킹함
        self.mask = (x <= 0)
        out = x.copy()
        # 마스크가 True인 위치의 값을 0으로 바꿈
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 순전파 시 0 이하였던 인덱스의 미분값을 0으로 차단함
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        # 순전파 출력값을 저장할 변수
        self.out = None

    def forward(self, x):
        # 시그모이드 수식을 계산하여 출력함
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # 상류 미분값에 순전파 출력값을 활용한 도함수 수식을 곱해 반환함
        dx = dout * (1.0 - self.out) * self.out
        return dx