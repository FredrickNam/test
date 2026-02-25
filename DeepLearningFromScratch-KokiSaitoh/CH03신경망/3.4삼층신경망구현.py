# 3.4 3층 신경망 구현
"""
1층(A)의 신호 전달 과정
- 구현 수식: $A = XW + B$
"""
import numpy as np

# 의존성 해결을 위한 활성화 함수 사전 정의
def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.array([1.,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X,W1) + B1

Z1 = sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
print(Z2)

def identity_function(x):
    return(x) # 출력층의 활성화 함수는 은닉층의 함수와는 다름을 표기

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)