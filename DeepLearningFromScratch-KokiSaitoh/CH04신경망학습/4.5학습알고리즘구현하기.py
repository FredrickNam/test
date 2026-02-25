# 4.5 학습알고리즘 구현하기
"""
신경망 학습의 Step
- 전제: 신경망에는 적응 가능한 가중치와 편향이 존재하며, 이를 훈련 데이터에 맞게 조정하는 과정이 학습임
- 1. 미니배치: 훈련 데이터 중 일부를 무작위로 가져옴 (이때 무작위 선정 방식을 '확률적 경사 하강법(SGD)'이라 부름)
- 2. 기울기 산출: 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구함
- 3. 매개변수 갱신: 가중치 매개변수를 기울기 방향으로 미세하게 갱신함
- 4. 반복: 1~3단계를 반복함

수치 미분 및 편미분 시 데이터 흐름
- numerical_gradient 내부 연산 시 가중치 배열 주소가 전달되어 가중치 값이 직접 변환됨
- 변환된 가중치에 대해 손실함수가 오차를 계산하고, 이를 기반으로 편미분 값을 도출함

과대적합(Overfitting)과 평가
- 훈련 데이터의 손실 함수 값이 작아지는 것만으로는 미지의 데이터에 대한 범용성을 보장할 수 없음
- 정기적으로 시험 데이터를 이용해 정확도를 측정하여 과대적합 발생 여부를 확인해야 함

학습 정리
- 목표: 손실함수가 가장 작아지는 가중치 매개변수를 찾는 것
- 훈련 데이터로 학습 후 시험 데이터로 범용 능력을 평가함
- 수치 미분은 구현이 간단하지만 실행 시간이 오래 걸리므로 오차역전파의 필요성이 대두됨
"""
import sys, os
import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,
                 output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1,W2 = self.params['W1'], self.params['W2']
        b1,b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y

    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

net = TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
print("W1:",net.params['W1'].shape)
print("b1:",net.params['b1'].shape)
print("W2:",net.params['W2'].shape)
print("b2:",net.params['b2'].shape)

(x_train, t_train),(x_test,t_test) = load_mnist(normalize = True, one_hot_label=True)

train_loss_list = []

iters_num = 10000 #반복 수
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
        loss=network.loss(x_batch,t_batch)
        train_loss_list.append(loss)

(x_train, t_train),(x_test,t_test) = load_mnist(normalize = True, one_hot_label=True)

train_loss_list = []

train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size/batch_size,1)

iters_num = 10000 #반복 수
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
        loss=network.loss(x_batch,t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train,t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc : "+str(train_acc),str(test_acc))