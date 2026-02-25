# 7.6 CNN시각화하기
"""
필터의 특징 시각화
- 학습을 완료한 첫 번째 계층의 합성곱 필터(가중치)를 이미지로 출력하면 흑백의 규칙성을 보임
- 원시 데이터에서 색상이 뚜렷하게 바뀌는 경계선(에지)이나 국소적으로 덩어리진 특정 영역(블롭)의 시각적 패턴을 감지하고 있음을 확인함 

층 깊이에 따른 정보 추출 양상 변화
- 계층이 겹겹이 쌓여 깊어질수록 추출되는 정보와 강하게 반응하는 뉴런의 성격이 고차원적으로 추상화됨
- 얕은 층에서는 단순한 에지를, 깊은 층에서는 특정 사물의 의미적 텍스처나 구조에 반응하게 됨
"""
import numpy as np
import matplotlib.pyplot as plt

def filter_show(filters, nx=8, margin=3, scale=10):
    # 전달받은 필터 4차원 배열의 형태를 분해함
    FN, C, FH, FW = filters.shape
    # 한 줄에 몇 개를 그릴지 지정된 nx 값을 기준으로 y축 방향 줄 수를 계산함
    ny = int(np.ceil(FN / nx))

    # 빈 캔버스를 생성하고 이미지 사이의 여백 공간을 조정함
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # 전체 필터 개수만큼 순회하며 개별 필터의 가중치를 흑백 이미지로 도화지에 그림
    for i in range(FN):
        # 좌표 눈금 표시를 없애고 i 번째 격자칸에 자리를 할당함
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        # 배열의 0번 채널 데이터를 흑백 반전 맵(gray_r)으로 표현하여 렌더링함
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    
    # 캔버스 결과를 화면에 띄움
    plt.show()

# 학습을 마친 네트워크의 첫 번째 필터(W1) 가중치를 전달하여 결과를 시각화함
filter_show(network.params['W1'])