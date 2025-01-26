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

# -- 矩阵实现单层神经网络
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],
              [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A = np.dot(X, W1) + B1
print(A) # [0.3 0.7 1.1]

sigmoid(A)

# -- 简单实现3层神经网络
def init_network():
    ''' 网络结构为 1*2 -> 1*3 -> 1*2 -> 1*2 '''
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],
                              [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4],
                              [0.2, 0.5],
                              [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3],
                              [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # A = XW + B
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W1) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W1) + b3
    y = a3

    return y


def softmax(x):
    max_x = np.max(x)  # 提取最大值
    new_x = x - max_x
    exp_new_x = np.exp(new_x)

    sum_exp_x = np.sum(exp_new_x)  # np.sum 计算矩阵各元素的总和
    y = exp_new_x / sum_exp_x

    return y

