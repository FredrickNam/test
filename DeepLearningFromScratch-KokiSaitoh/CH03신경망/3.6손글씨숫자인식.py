# 3.6 손글씨 숫자 인식
"""
추론(Predict) 처리 과정
- 이미 적절하게 설정된 가중치 w 와 편향 b 가 들어있는 pickle 파일(.pkl)을 그대로 불러와 사용함 (추론 과정만 구현)

배치(Batch) 처리 과정
- 여러 개의 입력 데이터를 하나의 배치로 묶어서 처리하는 방식 적용
"""
import sys, os
import numpy as np
import pickle
from PIL import Image

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# 의존성 해결을 위한 활성화 함수 사전 정의
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
img = x_train[0]
label = t_train[0]
print("img shape :",img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False, one_hot_label = False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt+=1
print("Accuracy:"+str(float(accuracy_cnt) / len(x)))

# 배치처리 과정
x,_ = get_data()
network = init_network()
W1,W2,W3 = network['W1'], network['W2'], network['W3']
print(x.shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p =np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))