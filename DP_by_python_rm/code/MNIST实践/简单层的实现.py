import numpy as np
import sys, os
from DP_by_python_rm.code.common.layers import MulLayer, AddLayer
sys.path.append(os.pardir)


# 输入层 苹果个数、橘子个数、苹果和橘子的价格、消费税
apple_cost = 100
orange_cost = 150
apple_num = 2
orange_num = 3
tax = 1.1

# 各层的初始化构建
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple_cost, apple_num) # (1)
orange_price = mul_orange_layer.forward(orange_cost, orange_num) # (2)
all_price = add_layer.forward(apple_price, orange_price) # (3)
price = mul_tax_layer.forward(all_price, tax) #

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_layer.backward(dall_price)


dapple_cost, dapple_number = mul_apple_layer.backward(dapple_price)
dorange_cost, dorange_number = mul_orange_layer.backward(dorange_price)


print(price)
print(dtax, dapple_cost, dorange_cost, dorange_number, dapple_number)


class ReLU():
    def __init__(self):
        self.mask = None # 前后向都需要的掩膜

    def forward(self, x):
        out = x.copy()
        self.mask = (x <= 0)
        out[self.mask] = 0

        return 0

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return 0
