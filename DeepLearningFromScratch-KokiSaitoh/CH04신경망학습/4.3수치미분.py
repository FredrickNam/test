# 4.3 수치미분
"""
수치미분의 문제점
- $ \frac{f(x + h) - f(x)}{h} $ 로 미분 구현 시 두 가지 문제가 존재함
- 반올림 오차: 작은 값을 무시하므로 h->0으로 수렴하기 힘듦
- 차분 오차: 점 $x+h$ 와 점 $x$ 사이의 기울기를 구한 것이지 실제 미분계수가 아니므로 오차가 발생함

중심 차분
- 중심 차분을 통해 오차를 줄임
- 수식: $ \frac{f(x + h) - f(x - h)}{2h} $

편미분
- 수식 $ f(x_0,x_1) = x_{0}^2 + x_{1}^2 $ 에서 변수가 2개이므로 각각의 변수에 대한 미분에 주의해야 함

함수 사용 예문
- $ x_0 = 3, x_1 = 4 $ 일 때, $ x_0 $ 및 $ x_1 $에 대한 편미분 과정을 보여줌
"""
def numerical_difference(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

#수치 미분의 예
def function_1(x):
    return 0.01*x**2 + 0.1*x

numerical_difference(function_1, 5)

def function_2(x):
    return x[0]**2+x[1]**2

# Step 1 : 상수 고정
def function_tmp1(x0): 
    return x0*x0 + 4.0**2.0

# Step 2 : x0을 기준으로 미분 진행
numerical_difference(function_tmp1,3)

# Step 1 : 상수 고정
def function_tmp2(x1): 
    return 3.0**2.0 + x1*x1

# Step 2 : x0을 기준으로 미분 진행
numerical_difference(function_tmp2,4)