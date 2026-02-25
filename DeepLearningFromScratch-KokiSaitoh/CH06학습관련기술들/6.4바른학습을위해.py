# 6.4 바른학습을 위해
"""
과대적합(Overfitting)
- 신경망이 훈련 데이터에 지나치게 적응하여 처음 보는 데이터(미지의 데이터)에는 제대로 대응하지 못하는 상태를 뜻함
- 매개변수가 많고 표현력이 높거나, 훈련 데이터가 지나치게 적을 때 주로 발생함

가중치 감소(Weight Decay)
- 학습 과정에서 큰 가중치에 상응하는 페널티를 부과하여 과대적합을 억제하는 수학적 기법임
- 손실 함수에 L2 노름(Norm) 페널티 항을 더해 가중치가 무분별하게 커지는 현상을 원천 방지함
- 가중치 $W$에 대한 L2 노름 감소 수식: $$\frac{1}{2} \lambda W^2$$
- L2 노름 수식: $$ \sqrt{w_1^2 + w_2^2 + ... w_n^2} $$
- 수식의 $W^2$는 가중치 원소들의 제곱의 합(Squared L2 Norm)을 의미하며, $\lambda$ 상수를 키울수록 페널티가 강하게 적용됨

드롭아웃(Dropout)
- 신경망 모델이 특정 뉴런 묶음에 과도하게 의존하는 것을 막기 위해 학습 시 뉴런을 임의로 삭제하는 정규화 기법임
- 훈련(Train) 시 은닉층의 뉴런을 무작위 확률로 0으로 만들어 신호 전달을 중간에 차단함
- 시험(Test) 시에는 모든 뉴런의 신호를 전달하되, 훈련 때 삭제하지 않은 비율을 곱하여 출력 균형 스케일을 조정함
- 여러 모델의 출력을 결합하여 평균 내는 앙상블(Ensemble) 학습과 알고리즘적으로 유사한 효과를 가짐
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

# 부모 디렉터리의 모듈 참조를 위해 환경 경로 추가함
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 1. 과대적합을 고의로 유발하는 실험
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 훈련 데이터 수를 300개로 극단적으로 줄여 과대적합 환경을 조성함
x_train = x_train[:300]
t_train = t_train[:300]

# 매개변수가 많은 6층 다층 신경망 모델을 생성함
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01)
max_epochs = 201

train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

epoch_cnt = 0
for i in range(10000):
    # 미니배치 무작위 데이터 추출함
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기를 산출하고 매개변수 갱신함
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    # 1 에폭마다 훈련 정확도와 시험 정확도 편차를 기록함
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 과대적합 추이 그래프 시각화함 (Train과 Test 그래프 이격 관찰)
epochs = np.arange(len(train_acc_list))
plt.plot(epochs, train_acc_list, label='train', color='blue')
plt.plot(epochs, test_acc_list, label='test', linestyle='--', color='orange')
plt.xlabel("epochs")
plt.ylabel("accuracy")    
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 2. 가중치 감소(L2 노름)를 적용한 실험
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = x_train[:300]
t_train = t_train[:300]

# weight_decay_lambda 매개변수 값을 0.1로 주어 L2 페널티 기능을 활성화함
network = MultiLayerNet(input_size=784, 
                        hidden_size_list=[100, 100, 100, 100, 100, 100], 
                        output_size=10,
                        weight_decay_lambda=0.1)

optimizer = SGD(lr=0.01)
max_epochs = 201

train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

epoch_cnt = 0
for i in range(100000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 가중치 감소 적용 후 정확도 그래프 시각화함 (오버피팅 간격 축소 확인)
epochs = np.arange(len(train_acc_list))
plt.plot(epochs, train_acc_list, label='train (with L2)', color='blue')
plt.plot(epochs, test_acc_list, label='test (with L2)', linestyle='--', color='orange')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 3. 드롭아웃(Dropout) 계층 클래스 구현
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        # 무작위 삭제 비율을 변수에 저장함
        self.dropout_ratio = dropout_ratio
        # 삭제할 뉴런 위치를 기록할 불리언 마스크 변수 초기화함
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # 훈련 시, 입력 데이터와 동일한 형상의 난수 배열 생성 후 삭제 비율보다 큰 요소만 True로 보존함
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 시험 시, 훈련 때 살아남은 비율만큼 곱하여 전체 출력 스케일 균형을 맞춤
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 역전파 시, 순전파에서 통과시킨 위치(마스크가 True인 요소)만 미분값을 하류로 흘림
        return dout * self.mask