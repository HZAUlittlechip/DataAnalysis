import numpy as np

# 均方误差的实现
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 不同输出y的均方差区别，来个小例子
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
result = mean_squared_error(y, t)
print(result)

# 交叉墒误差的实现
def cross_entropy_error_1(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
result = cross_entropy_error_1(y, t)
print(result)

y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
result = cross_entropy_error_1(y, t)
print(result)

# mini_batch 版交叉熵one-hot标签版
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 转为2维向量
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 导数函数
def numercial_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h) # 主要分子成为2h

# 偏导函数的计算演示
def function_tmp1(x0): # 展示的函数表达
    return x0 * x0 + 4.0**2.0

numercial_diff(function_tmp1, 3.0) # 调用前面的导数求解函数
print(numercial_diff(function_tmp1, 3.0))

# 梯度函数 *
def numerical_gradient_1(f, x):
    """ 输入 处理函数f 和 输入量x 输出梯度"""
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


# 梯度下降实现 *
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """ 梯度下降过程模拟，输入 函数和输入量，内部自动计算梯度进行梯度下降迭代"""
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient_1(f, x)
        x -= lr * grad

    return x

# 梯度下降过程的模拟，求函数的最小值
def function_2(x):
    return np.sum(x ** 2)

init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(result) # [-6.11110793e-10  8.14814391e-10]

# 简单神经网络的构建
""" 需要函数； softmax, cross_entropy_error, numerical_gradient"""
def softmax(x):
    max_x = np.max(x)
    new_x = x - max_x
    exp_new_x = np.exp(new_x)

    sum_exp_x = np.sum(exp_new_x)
    y = exp_new_x / sum_exp_x

    return y
# *
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 随机用高斯分布生成权重矩阵

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

# 给个事例看看
net = simpleNet()
print(net.W) # 随机生成的权重矩阵

x = np.array([0.6, 0.9])
p = net.predict(x) # 其实在这个生成的过程中，我们是看不到权重的生成过程的，这可能就是封装吧
print(p) # 类似与输出值前项节点a

np.argmax(p) # 获取最大值的索引
print(np.argmax(p))

t = np.array([0, 0, 1]) # 正确解的标签
net.loss(x ,t)
print(net.loss(x ,t))

def f(W):
    return net.loss(x, t) # 这里定义的f（W）其实说明的损失函数与权重有关，虽然其需要输入的是x 和 t
# f = lambda w: net.loss(x, t) 和上面的函数等价

dW = numerical_gradient_2(f, net.W) # 需要使用遍历2维数组版的数值梯度计算函数
print(dW)




















