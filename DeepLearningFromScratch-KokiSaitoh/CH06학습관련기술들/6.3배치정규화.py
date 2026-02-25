# 6.3 배치 정규화
"""
배치 정규화(Batch Normalization)의 개념
- 가중치 초깃값에 의존하지 않고 각 층의 활성화 분포가 적당히 퍼지도록 학습 과정 자체에서 강제하는 기법임
- Affine 계층과 ReLU 계층 사이에 배치 정규화 계층(BatchNorm)을 삽입하여 데이터 분포를 층간 이동 시마다 정규화함

배치 정규화의 장점
- 학습 속도를 대폭 개선할 수 있음
- 가중치 초깃값 선택에 대한 의존도가 낮아짐
- 과대적합(Overfitting)을 억제하는 부수적 효과가 있음

배치 정규화 수식 (미니배치 기준)
- 미니배치 평균 (Mean):
  $$\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i$$
- 미니배치 분산 (Variance):
  $$\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$
- 데이터 정규화 (Normalize): 평균이 0, 분산이 1이 되도록 변환 (0으로 나누는 것 방지 위해 $\epsilon$ 추가함)
  $$\hat{x}_i \leftarrow \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
- 확대 및 이동 변환: 정규화된 데이터에 고유한 스케일과 이동을 적용함
  $$y_i = \gamma \hat{x}_i + \beta$$
- $\gamma$(확대)는 1, $\beta$(이동)는 0부터 시작하여 학습 과정에서 역전파를 통해 적절한 값으로 자율 갱신됨
"""