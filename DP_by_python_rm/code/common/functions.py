""" 存储计算函数"""
import numpy as np

# 与门
def AND(x1, x2):
    # 权重和阈值设置（0.5，0.5，0.7）
    w1, w2, delta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp > delta:
        return 1
    else:
        return 0

# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.6
    tmp = sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0
    tmp = sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0

# 异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# 阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(np.int32)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 有溢出的可能


def sigmoid_1(x):
    # 如果x是数组，使用np.any()或np.all()来避免比较整个数组
    if np.any(x > 1000):
        return 1
    elif np.any(x < -1000):
        return 0
    else:
        return 1 / (1 + np.exp(-x))



# ReLU函数
def relu(x):
    return np.maximum(0, x)

# 恒等函数
def identity_function(x):
    return x

# softmax函数
def softmax(x):
    max_x = np.max(x)  # 提取最大值
    new_x = x - max_x
    exp_new_x = np.exp(new_x)
    sum_exp_x = np.sum(exp_new_x)  # np.sum 计算矩阵各元素的总和
    y = exp_new_x / sum_exp_x

    return y

# 均方误差（MSE）
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error_onehot(y, t):
    """ mini_batch版 """
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 转为2维向量
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_pure(y, t):
    """  测试集非ont-hot模式"""
    if y.ndim == 1:
        t = t.reshape(1, t.size) # 转为2维向量
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size

# 数值微分
def numercial_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h) # 主要分子成为2h

# 梯度计算
def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad

# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """ 可修改学习率lr 和 步数step_num"""
    x = init_x

    for i in range(step_num):
        grad = _numerical_gradient_1d(f, x)
        x -= lr * grad

    return x