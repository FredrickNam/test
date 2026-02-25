# 7.5 CNN구현하기
"""
CNN 구조 조립 및 학습 적용
- 합성곱 계층(Conv)과 풀링 계층(Pooling)을 조합하여 손글씨 인식 신경망(SimpleConvNet)을 구성함
- 구성 흐름: Conv -> ReLU -> Pooling -> Affine -> ReLU -> Affine -> Softmax
- 데이터 처리 요약: 합성곱 계층이 이미지의 특징을 유지하며 압축시키고, 풀링 계층으로 핵심 정보만 뽑아낸 다음 마지막 완전연결 계층(Affine)이 특징을 바탕으로 최종 분류(예측)를 수행함
- 즉, CNN은 이미지의 대표적 특징을 효과적으로 추상화하여 뽑아내주는 역할을 수행함
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict

sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from common.trainer import Trainer

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        # 딕셔너리로 받은 합성곱 설정 매개변수를 꺼내어 지역 변수로 할당함
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        
        # 합성곱 계층과 풀링 계층을 통과한 후의 출력 형상을 계산함
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 3개의 주요 계층(Conv1, Affine1, Affine2)에 사용할 가중치와 편향을 초기화함
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 순서를 보장하는 딕셔너리(OrderedDict)를 사용해 신경망 계층을 구성함
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], 
                                           self.params['b1'],
                                           conv_param['stride'], 
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        # 마지막 출력단의 손실 함수 계층을 개별적으로 저장함
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 딕셔너리에 등록된 각 계층의 순전파 메서드를 차례로 호출하여 예측함
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        # 예측값 도출 후 마지막 계층에서 오차(Loss)를 계산함
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        # 정답 레이블이 원-핫 인코딩 형태일 경우 인덱스 값으로 변환함
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        
        # 메모리 효율을 위해 미니배치 단위로 쪼개어 정확도 연산을 반복 수행함
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            # 예측 인덱스와 정답 인덱스가 같은 것들의 수를 누적함
            acc += np.sum(y == tt) 
        
        # 전체 데이터 개수로 나누어 정확도 비율을 반환함
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        # 전통적이고 느린 수치 미분 방식으로 매개변수의 기울기를 도출함
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        # 오차역전파법의 시작을 위해 먼저 순전파 연산을 발생시킴
        self.loss(x, t)

        # 마지막 계층에서 역전파 신호(1)를 하류로 흘려보냄
        dout = 1
        dout = self.last_layer.backward(dout)

        # 저장된 계층 딕셔너리를 리스트로 전환 후 역순으로 뒤집음
        layers = list(self.layers.values())
        layers.reverse()
        
        # 역방향으로 각 계층의 backward를 호출하여 고속으로 기울기를 구함
        for layer in layers:
            dout = layer.backward(dout)

        # 연산 결과 저장된 각 계층의 도함수 값들을 딕셔너리로 취합하여 반환함
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

# 학습 스크립트 실행부
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 빠른 연산 테스트를 위해 데이터의 일부만 슬라이싱하여 추출함
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

# 설계한 CNN 모델 인스턴스화 수행함
network = SimpleConvNet(input_dim=(1, 28, 28), 
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
# 외부 공통 모듈인 Trainer를 사용해 번거로운 학습 과정 처리를 위임함 (최적화 기법: Adam)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 학습 완료 후 에포크별 정확도 추이를 시각화하여 확인하는 로직임
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()