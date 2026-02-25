# 6.2 가중치 초깃값
"""
가중치 감소 기법과 초깃값의 중요성
- 가중치 감소: 가중치 매개변수 값이 작아지도록 억제하여 과대적합을 방지하고 범용성을 높이는 기법임
- 가중치 초깃값을 0으로 설정하면 오차역전파 시 모든 가중치가 동일하게 갱신되므로 대칭성 파괴를 위해 무작위 값을 사용해야 함

은닉층의 활성화값 분포와 기울기 소실
- 가중치의 표준편차를 1로 설정할 경우, 노드가 많을 때 활성화 값이 0과 1로 치우치며 시그모이드 함수의 미분값이 0에 수렴하는 '기울기 소실(Gradient Vanishing)' 현상이 발생함
- 가중치의 표준편차를 0.01로 설정할 경우, 기울기 소실은 없으나 활성화 값이 0.5 부근에 밀집하여 다수의 뉴런이 동일한 값을 출력하므로 '표현력 제한' 문제가 발생함

Xavier 초깃값
- 딥러닝 프레임워크의 표준적인 초깃값임
- 앞 계층의 노드가 $n$개일 때, 활성화 값을 광범위하게 분포시키기 위해 표준편차가 $\frac{1}{\sqrt{n}}$인 정규분포를 사용함
- 시그모이드나 tanh처럼 좌우 대칭이고 중앙이 선형인 활성화 함수에 적합함

He 초깃값
- ReLU 함수에 특화된 가중치 초깃값임
- 앞 계층의 노드가 $n$개일 때, 표준편차가 $\sqrt{\frac{2}{n}}$인 정규분포를 사용함
- ReLU 사용 시 Xavier 초깃값을 적용하면 층이 깊어질수록 치우침이 커져 기울기 소실이 발생하므로 반드시 He 초깃값을 사용해야 함
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # 시그모이드 활성화 함수 수식을 계산하여 반환함
    return 1 / (1 + np.exp(-x))

# 1000개의 데이터를 100차원으로 무작위 생성함
x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

# 1. 표준편차 1을 사용한 가중치 초깃값 실험
for i in range(hidden_layer_size):
    if i != 0:
        # 이전 층의 활성화 값을 현재 층의 입력으로 이어받음
        x = activations[i-1]
    # 표준편차가 1인 정규분포를 사용하여 가중치 행렬 생성함
    w = np.random.randn(node_num, node_num) * 1
    # 행렬 내적을 통해 가중치 총합 계산함
    a = np.dot(x, w)
    # 시그모이드 함수를 통과시켜 활성화 값 도출함
    z = sigmoid(a)
    activations[i] = z

# 각 층의 활성화 값 히스토그램을 그려 기울기 소실 현상 확인 점검함
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

# 2. 표준편차 0.01을 사용한 가중치 초깃값 실험 (표현력 제한 문제 발생)
x = np.random.randn(1000, 100)
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # 가중치 초깃값의 표준편차를 0.01로 대폭 축소함
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

# 3. Xavier 초깃값을 사용한 실험 (활성화 값의 고른 분포 달성)
x = np.random.randn(1000, 100)
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # 앞 노드 개수(node_num)의 제곱근 역수를 표준편차로 사용하여 가중치 생성함 (Xavier 초깃값)
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()