# 7.4 합성곱 풀링 계층 구현
"""
4차원 데이터와 im2col 전개
- CNN에서 흐르는 데이터는 4차원(데이터 개수, 채널, 높이, 너비) 배열 형태임
- 합성곱 연산을 for문으로 구현하면 중첩이 많아 느려지므로 데이터를 평탄하게 펴는 im2col 함수를 사용하여 전개함
- 필터를 적용하는 블록을 한 줄로 늘어놓아 행렬 계산 라이브러리의 빠른 속도를 이끌어냄

Convolution(합성곱 계층) 구현
- 전개된 입력 행렬과 세로로 펴진 필터 가중치 행렬을 내적(np.dot) 연산함
- 연산이 끝난 2차원 결과를 다시 원래의 4차원 구조로 재배열(reshape)함

Pooling(풀링 계층) 구현
- 합성곱 계층과 마찬가지로 im2col을 이용해 입력 영역을 전개함
- 풀링은 채널마다 독립적으로 전개되며, 각 행에서 최댓값(np.max)을 뽑아내어 공간 크기만 압축함
"""
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # 입력 4차원 배열에서 각 차원 크기 정보를 변수로 분리함
    N, C, H, W = input_data.shape
    # 스트라이드와 패딩 공식을 통해 출력 결과물의 높이와 너비를 사전 계산함
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 공간적 크기(H, W)의 상하좌우에만 지정된 pad 크기만큼 0을 채워 넣음
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # 나중에 깔끔하게 전개하기 위해 데이터를 임시 보관할 6차원 0 배열을 준비함
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 필터 크기만큼 반복하며 슬라이싱을 이용해 필터가 닿는 모든 위치의 데이터를 복사해 옴
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # 파이썬 슬라이싱의 꼼수를 이용해 for문을 최소화하여 연산 속도를 확보함
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 임시 배열의 축을 섞은 뒤 2차원 행렬로 완전히 평탄화하여 반환함 (-1은 나머지 원소 크기에 맞춤)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

class Convolution:
    def __init__(self, W, b, stride=1, pad=1):
        # 가중치(필터), 편향, 스트라이드, 패딩 값을 인스턴스에 기록함
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # 가중치와 입력 데이터의 형태를 파악함
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 합성곱 결과의 높이와 너비 값을 수식대로 계산함
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 4차원 입력 데이터를 행렬 곱셈을 위해 거대한 2차원 표(행렬)로 펴버림
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 4차원 필터 뭉치를 세로로 길쭉한 1차원 막대기로 편 다음 전치하여 행으로 세움
        col_W = self.W.reshape(FN, -1).T
        
        # 펴진 이미지 배열과 펴진 필터 배열을 한 번의 행렬 내적 연산으로 곱하고 편향을 더함
        out = np.dot(col, col_W) + self.b

        # 행렬 연산 결과를 다시 4차원 텐서 형태로 복원함
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        # 풀링 적용 영역의 높이/너비와 스트라이드 정보를 기록함
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        # 풀링 출력 크기 계산함
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        # 입력 데이터를 전개함
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 채널 정보가 섞이지 않도록 전개된 행렬의 형태를 수정함
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 전개된 영역을 행별로 탐색하여 가장 큰 값을 추출함
        out = np.max(col, axis=1) 
        
        # 1차원이 된 최댓값 배열을 적절한 4차원 구조로 변환함
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

# 데이터 전개 테스트를 위한 무작위 배열 세팅 및 크기 확인 출력부
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)