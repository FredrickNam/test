# 3.3 다차원 배열의 계산
"""
다차원 배열 기본 메서드
- np.ndim(A): A의 차원을 정수로 반환함
- A.shape: A의 형상을 튜플로 반환함 (m x n 배열 구조를 의미)

행렬의 도트 곱
- np.dot(A,B) 함수 사용
- 행렬의 곱을 구현한 것이라 규칙 같음
- 수식: $(a)ik * (b)lj = (c)ij$
- 앞 행렬의 열 개수와 뒤 행렬의 행 개수가 일치해야 연산 가능 (m x r * r x n = m * n 꼴)

신경망에서의 행렬 곱
- X (입력값의 행렬), W (가중치의 행렬), Y (결괏값의 행렬)
- 수식 $X W = Y$ 형태로 생각 가능
"""
import numpy as np

A = np.array([[1.,2.,3.],[2.,4.,5.]])
print(np.ndim(A))
print(A.shape)

B = np.array([[1,3],[2,3],[4,5]])
print(np.dot(A,B))

X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
Y = np.dot(X,W)
print(Y)