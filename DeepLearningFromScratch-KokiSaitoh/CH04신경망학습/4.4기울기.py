# 4.4 기울기
"""
기울기 (Gradient)
- 변수별 편미분을 동시에 계산하기 위해 모든 변수의 편미분을 벡터로 정리한 것을 기울기라 함
- 수식 예: $(\frac{\partial f}{\partial x_0},\frac{\partial f}{\partial x_1})$
- 기울기는 함수의 가장 낮은 값을 가리키는 방향(출력값을 가장 크게 줄이는 방향)을 나타냄

기울기 동작에 대한 시각적 묘사 및 비유 설명
- 수식 $z = x_{0}^2 + x_{1}^2$ 그래프를 아래쪽이 둥근 '밥그릇' 모양의 산으로 비유하여 묘사함
- 현재 위치에서 기울기 벡터(Gradient)는 원점의 정반대 방향(가장 가파르게 높아지는 바깥쪽)을 향하는 화살표 형태를 띰
- 하지만 학습 과정에서는 이 기울기 벡터에 마이너스(-)를 붙여 산의 내리막길(가장 낮은 값) 방향으로 이동하게 됨을 상세히 서술함

경사하강법 (Gradient Descent)
- 최적의 매개변수(손실함수가 최솟값이 될 때의 가중치와 편향)를 찾는 과정
- 기울어진 방향으로 일정 거리만큼 이동을 반복하며 최솟값을 찾음
- 단, 기울어진 방향이 반드시 최솟값을 가리키는 것은 아니며 안장점(saddle point)에 빠질 위험이 존재함
- 수식:
  $$x_{0} = x_{0} - \eta \frac{\partial f}{\partial x}$$
  $$x_{1} = x_{1} - \eta \frac{\partial f}{\partial x}$$
- $ \eta \text{  에타}^{eta} $ 기호는 갱신하는 양으로 '학습률'이라 부름

경사법 특징
- 변수의 수가 늘어나도 동일한 식 사용 가능
- 학습률은 미리 정해 두어야 하며 너무 크거나 작으면 최적의 장소를 찾지 못함

신경망에서의 기울기
- 가중치 매개변수에 대한 손실함수의 기울기를 구해야 함
- 기울기 행렬의 형상은 가중치 행렬 W와 동일하여 연산이 가능함
- 수식:
  $$Gradient = \frac{\partial L}{\partial W} \text{  (L: LossFunction, W: Weight)}$$
  $$W = \begin{pmatrix} 
  W_{11} & W_{12} & W_{13} \\ 
  W_{21} & W_{22} & W_{23} 
  \end{pmatrix}$$
  $$\frac{\partial L}{\partial W} = \begin{pmatrix} 
  \dfrac{\partial L}{\partial W_{11}} & \dfrac{\partial L}{\partial W_{12}} & \dfrac{\partial L}{\partial W_{13}} \\ 
  \dfrac{\partial L}{\partial W_{21}} & \dfrac{\partial L}{\partial W_{22}} & \dfrac{\partial L}{\partial W_{23}} 
  \end{pmatrix}$$
"""
import sys, os
import numpy as np

sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

def function_2(x):
    return x[0]**2+x[1]**2

def numerical_gradient(f,x): #f에는 함수 x에는 array 받음
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad

numerical_gradient(function_2, np.array([3.0,4.0]))

def gradient_descent(f, init_x, lr=0.01,step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x) #앞서 작성한 편미분 식
        x -= lr*grad
    return x

init_x = np.array([3.0,-4.4])
gradient_descent(function_2,init_x = init_x, lr = 0.01, step_num = 1000)

init_x = np.array([3.0,-4.4])
print("학습률 너무 클 때 :",gradient_descent(function_2,init_x = init_x, lr = 10, step_num = 1000))
print("학습률 너무 작을 때 :", gradient_descent(function_2,init_x = init_x, lr = 1e-10, step_num = 1000))

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.7])
p = net.predict(x)
print(p)

t=np.array([0,0,1])
net.loss(x,t)

def f(W):
    return net.loss(x,t)

dW = numerical_gradient(f, net.W)
print(dW)