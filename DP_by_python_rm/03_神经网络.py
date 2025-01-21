# -- 阶跃函数的图形类型实现
import numpy as np
import matplotlib.pyplot as plt


# - 简单形式 但其无法输入np的数组形式
def step_function_origin(x):
    if x > 0:
        return 1
    else:
        return 0

# - 可对np数组函数进行处理
def step_function(x):
    y = x > 0
    return y.astype(np.int32) # 元素转化
# 解释
x = np.array([-1.0, 1.0, 2.0])
y = x > 0 # array([False,  True,  True], dtype=bool)
y.astype(np.int32) # array([0, 1, 1])

# - 阶级函数图的实现
x = np.arange(-5.0, 5.0, 0.1) # 均匀步长
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# -- sigmoid 函数实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1) # 均匀步长
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# -- ReLU函数实现
def relu(x):
    return np.maximum(0, x)