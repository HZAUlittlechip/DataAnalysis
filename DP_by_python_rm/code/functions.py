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
    return -np.sum(np.log(y[np.arange(batch_size),y] + 1e-7)) / batch_size

# 数值微分
def numercial_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h) # 主要分子成为2h

# 梯度计算
def numerical_gradient_1(f, x):
    """ 针对x为1维数组的情况 """
    h = 1e-4
    grad = np.zeros_like(x) # 生成和x形状相同的数组，数组内都是0

    for idx in range(x.size):
        tmp_val = x[idx] # 中间变量
        x[idx] = tmp_val + h # f(x + h)计算
        f_xh1 = f(x)

        x[idx] = tmp_val - h # f(x - h）计算
        f_xh2 = f(x)

        grad[idx] = (f_xh1 - f_xh2)/(h * 2)
        x[idx] = tmp_val # 还原x

    return grad

def numerical_gradient_2(f, x):
    """ 针对x为2维数组的情况 """
    h = 1e-4  # 偏移量
    grad = np.zeros_like(x)  # 初始化梯度数组

    # 针对二维数组的每个元素计算梯度
    for i in range(x.shape[0]):  # 遍历行
        for j in range(x.shape[1]):  # 遍历列
            tmp_val = x[i, j]

            # 计算f(x+h) 和 f(x-h)
            x[i, j] = tmp_val + h
            fxh1 = f(x)

            x[i, j] = tmp_val - h
            fxh2 = f(x)

            # 数值梯度
            grad[i, j] = (fxh1 - fxh2) / (2 * h)
            x[i, j] = tmp_val  # 恢复原始值

    return grad

# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """ 可修改学习率lr 和 步数step_num"""
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient_1(f, x)
        x -= lr * grad

    return x