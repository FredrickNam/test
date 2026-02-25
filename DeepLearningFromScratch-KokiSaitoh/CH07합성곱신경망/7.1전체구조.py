# 7.1 전체구조
"""
CNN(합성곱 신경망)의 특징
- 지금까지의 신경망처럼 계층을 쌓아 만들 수 있으나 합성곱 계층과 풀링 계층이 새롭게 등장함
- 목표: 여러 계층을 어떻게 조합하여 CNN을 만드는지 확인하는 것임

완전연결 계층(Fully Connected Layer)과의 비교
- 기존 방식: 인접하는 계층의 모든 뉴런이 결합되어 있는 형태임 (구현에서 Affine 계층에 해당함)
- 기존 흐름: Affine -> ReLU -> Affine -> Softmax
- 문제점: 데이터의 공간적 형상이 무시됨

CNN의 기본 구조 흐름
- CNN의 구조는 'Conv - ReLU - Pooling' 흐름으로 주로 연결됨 
- CNN의 흐름 예시: Conv -> ReLU -> Pooling -> Conv -> ReLU -> Affine -> ReLU -> Softmax
- 출력에 가까운 층에서는 기존의 Affine-ReLU 조합을 사용하고, 마지막 출력 계층에서는 Affine-Softmax 조합을 유지함
"""