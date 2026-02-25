# 5.4 단순한 계층 구현하기
"""
곱셈 계층과 덧셈 계층 구현
- 사과와 귤을 구매하는 쇼핑 예제를 각각의 노드 계층으로 분리하여 객체지향적으로 구현함
- 각 계층은 순전파(forward)와 역전파(backward) 메서드를 가짐
"""

class MulLayer:
    def __init__(self):
        # 순전파 시의 입력값을 유지하기 위한 변수 초기화
        self.x = None
        self.y = None

    def forward(self, x, y):
        # 입력값을 인스턴스 변수에 저장함
        self.x = x
        self.y = y
        # 두 입력값을 곱하여 순전파 결과 도출함
        out = x * y
        return out

    def backward(self, dout):
        # 상류에서 넘어온 미분값(dout)에 서로 바꾼 입력값을 곱하여 하류로 흘림
        dx = dout * self.y 
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        # 덧셈 노드는 역전파 시 입력값이 필요 없으므로 별도 저장하지 않음
        pass

    def forward(self, x, y):
        # 두 입력값을 단순히 더하여 출력함
        out = x + y
        return out

    def backward(self, dout):
        # 상류에서 넘어온 미분값을 그대로 1을 곱해 하류로 전달함
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# 사과와 귤 쇼핑 예제 데이터 설정
apple = 100
apple_num = 2
tangerine = 50
tangerine_num = 6
tax = 1.1

# 각 계산 과정에 해당하는 노드 계층 인스턴스 생성
mul_apple_layer = MulLayer()
mul_tangerine_layer = MulLayer()
add_apple_tangerine_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파 계산 실행 과정
apple_price = mul_apple_layer.forward(apple, apple_num)
tangerine_price = mul_tangerine_layer.forward(tangerine, tangerine_num)
price = add_apple_tangerine_layer.forward(apple_price, tangerine_price)
cost = mul_tax_layer.forward(price, tax)

# 역전파 계산 실행 과정
dprice = 1
# 최상류의 미분값 1을 시작으로 역방향 연산 수행함
dprice, dtax = mul_tax_layer.backward(dprice)
dapple_price, dtangerine_price = add_apple_tangerine_layer.backward(price)
dtangerine, dtangerine_num = mul_tangerine_layer.backward(dtangerine_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(cost)
print(dapple_num, dapple, dtangerine, dtangerine_num, dtax)