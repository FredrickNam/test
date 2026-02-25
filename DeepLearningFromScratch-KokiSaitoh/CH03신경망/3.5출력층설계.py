# 3.5 출력층 설계
"""
소프트맥스 함수 구현과 개선
- 지수 함수(e^x)를 활용하므로 x값이 커지면 오버플로(Overflow)가 발생함
- 오버플로를 방지하기 위해 로그 성질을 활용, 분자/분모에 임의의 실수값을 반영해 식을 변형할 수 있음
- 통상적으로 입력 배열 요소 중 최댓값을 빼주는 방식을 사용함

소프트맥스(Softmax) 함수의 특징
- 0에서 1.0 사이의 실수값을 출력함
- 출력의 총합이 1임 (확률로 해석이 가능해짐)
"""
import numpy as np

def softmax_(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

a = np.array([1010,1000,990])
print(softmax(a))