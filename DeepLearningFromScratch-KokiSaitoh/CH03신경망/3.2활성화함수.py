# 3.2 활성화 함수
"""
계단 함수
- 임곗값을 경계로 출력이 바뀜
- 앞서 본 퍼셉트론의 함수 또한 계단 함수임

시그모이드 함수
- 수식: h(x) = 1 / 1 + exp(-x)
- 신경망에서 자주 사용되는 함수임

두 활성 함수 비교
- 공통점
  - 입력이 작으면 0에 가까움(혹은 0)
  - 입력이 크면 1에 가까움(혹은 1)
  - 둘 다 비선형 함수임 (선형 함수는 층이 아무리 깊어도 층이 없는 네트워크와 동일하기에 활성화 함수로서 이점이 없음)
- 차이점
  - 시그모이드 함수: 입력 따라 매끄럽게 변함, 연속적인 값 반환
  - 계단 함수: 경계를 기준으로 급격하게 변함, 0 또는 1만 반환

ReLU 함수
- 최근에는 활성화 함수로 ReLU를 많이 사용함
"""
import numpy as np
import matplotlib.pyplot as plt

# 계단함수 구현 (실수 입력 ver)
def step_function_real(x):
    if x>0:
        return 1
    else:
        return 0

# 계단함수 구현 (넘파이 배열 입력ver)
def step_function(x):
    y = x > 0 # Bool 배열로 변환
    return y.astype(int) # astype로 true 는 1 flase 는 0 으로 바꿈

# 실사용
x_temp = np.array([-1.0, 1.0, 2.0])
print(step_function(x_temp))

# 계단함수의 그래프
x_step = np.arange(-5.0,5.0,0.1)
y_step = step_function(x_step)
plt.plot(x_step,y_step)
plt.ylim(-0.1,1.1)
plt.show()

# 시그모이드 함수 구현
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 실사용
x_temp = np.array([-1.,1.,2.])
print(sigmoid(x_temp))

# 시그모이드 함수의 그래프
x_sig = np.arange(-5.,5.,0.1)
y_sig = sigmoid(x_sig)
plt.plot(x_sig,y_sig)
plt.ylim(-0.1,1.1)
plt.show()

# 두 그래프 함께 그리기
plt.plot(x_step,y_step)
plt.plot(x_sig,y_sig)
plt.ylim(-0.1,1.1)
plt.show()

# ReLU 구현
def ReLU(x):
    np.maximum(0,x)