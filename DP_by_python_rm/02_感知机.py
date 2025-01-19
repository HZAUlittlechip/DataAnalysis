# -- 感知机的实现
# - 与门（And gate）
def AND(x1, x2):
    # 权重和阈值设置（0.5，0.5，0.7）
    w1, w2, delta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp > delta:
        print(1)
        return 1
    else:
        print(0)
        return 0

AND(1, 1) # 1
AND(1, 0) # 0
AND(0, 1) # 0

import numpy as np
# -- 与门的实现
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

# -- 与非门（NAND）的实现
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.6
    tmp = sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

# -- 或门（OR）的实现
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0
    tmp = sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

# -- 利用多层感知机实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y