# 5.3 역전파
"""
덧셈 노드의 역전파
- $z = x + y$ 수식 기반의 덧셈 노드 역전파 과정
- 수식: $\frac{\partial z}{\partial x} = 1$, $\frac{\partial z}{\partial y} = 1$
- 상류에서 전해진 미분값에 1을 곱하므로, 덧셈 노드는 입력 신호를 변형 없이 그대로 하류 노드로 전달함

곱셈 노드의 역전파
- $z = xy$ 수식 기반의 곱셈 노드 역전파 과정
- 수식: $\frac{\partial z}{\partial x} = y$, $\frac{\partial z}{\partial y} = x$
- 상류에서 전해진 미분값에 순전파 때의 입력값을 서로 바꾸어 곱한 뒤 하류 노드로 전달함
"""