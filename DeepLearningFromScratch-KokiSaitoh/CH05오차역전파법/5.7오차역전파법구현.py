# 5.7 오차역전파법 구현
"""
오차역전파법을 활용한 신경망 학습 구현
- Step 1 (미니배치): 훈련 데이터의 일부를 무작위로 추출하여 학습 단위를 구성함
- Step 2 (기울기 산출): 각 가중치 매개변수에 대해 오차를 가장 작게 하는 기울기 방향을 오차역전파법으로 고속 산출함
- Step 3 (매개변수 갱신): 구해진 기울기를 통해 가중치를 학습률만큼 미세 조정함
- Step 4 (반복): 위 과정을 정해진 횟수만큼 반복하며 모델을 최적화함

신경망 클래스 (TwoLayerNet) 구조
- 계층(Layer) 기반 아키텍처: 순서가 보장되는 OrderedDict 객체에 Affine, Relu 등의 계층을 순서대로 담아 처리 과정을 모듈화함
- 순전파는 등록된 계층의 forward를 순차적으로 호출하여 간결하게 진행됨
- 역전파는 계층 리스트를 역순으로 순회하며 backward를 호출하여 기울기를 쉽게 구함

기울기 검증 (Gradient Check)
- 수치 미분: 느리지만 로직이 직관적이라 구현 실수가 적음
- 오차역전파법: 연쇄법칙으로 고속 계산하지만 수식 구현 시 오류 발생 가능성이 있음
- 두 방식의 결과를 비교하여 오차역전파 로직이 올바르게 구현되었는지 검증하는 과정을 포함함
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict

# 사전 필요 모듈 임포트 (독립적 실행을 위해 명시함)
from dataset.mnist import load_mnist
from common.gradient import numerical_gradient
# 앞서 구현한 레이어들을 사용한다고 가정 (코드 통합성을 위해 생략 없이 활용)
# 본 코드 상단에 Affine, Relu, SoftmaxWithLoss 클래스가 정의되어 있어야 정상 작동함

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 및 편향 매개변수 초기화 설정함
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # OrderedDict를 사용하여 신경망 계층을 순서대로 보관함
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 마지막 계층인 손실 함수 계층 등록함
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # 딕셔너리에 저장된 각 계층의 forward를 차례로 호출하여 예측 수행함
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        # 예측 수행 후 마지막 손실 계층을 통해 최종 오차값 계산함
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # 예측 확률 중 가장 높은 값의 인덱스를 추출하여 정답과 비교함
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        # 원-핫 인코딩 형태일 경우 정답 인덱스로 변환함
        if t.ndim != 1: 
            t = np.argmax(t, axis=1) 

        # 전체 데이터 중 정답을 맞춘 비율을 계산함
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        # 수치 미분을 사용한 전통적이고 느린 기울기 산출 방식임 (검증용)
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # 오차역전파법을 이용한 고속 기울기 산출 방식임
        
        # 1. 순전파를 통해 각 계층의 중간 데이터를 메모리에 기록함
        self.loss(x, t) 

        # 2. 마지막 손실 계층부터 1의 미분값을 시작으로 역방향 전파를 시작함
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        # 계층 딕셔너리를 리스트로 변환 후 순서를 뒤집어 역순으로 순회함
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 3. 역전파를 통해 구해진 각 Affine 계층의 기울기를 딕셔너리로 묶어 반환함
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

# --- 1. 기울기 검증 구현부 ---
# 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 신경망 인스턴스 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 배치 데이터 3개만 추출하여 테스트 진행함
x_batch = x_train[:3]
t_batch = t_train[:3]

# 두 가지 방식으로 각각 기울기를 도출함
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 매개변수 기울기의 절대 오차 평균을 구하여 차이가 거의 없는지 확인 (기울기 확인 과정)
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

# --- 2. 실제 학습 루프 구현부 ---
# 전체 변수 초기화 재설정 (독립적 맥락 유지)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1 에폭(Epoch)당 반복 횟수 계산함
iter_per_epoch = max(train_size / batch_size, 1)

# 설정된 반복 횟수만큼 학습 진행함
for i in range(iters_num):
    # 미니배치 데이터 무작위 추출함
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법을 활용해 기울기를 고속으로 산출함
    grad = network.gradient(x_batch, t_batch)

    # 산출된 기울기와 학습률을 곱해 기존 가중치 매개변수를 갱신함
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 확인을 위해 오차값을 기록함
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 에폭이 끝날 때마다 훈련 데이터와 시험 데이터의 정확도를 측정하여 과적합 여부를 모니터링함
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)