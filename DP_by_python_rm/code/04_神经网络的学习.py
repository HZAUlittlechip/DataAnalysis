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
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
result = cross_entropy_error(y, t)
print(result)

y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
result = cross_entropy_error(y, t)
print(result)
# mini-batch（小批量数据抓取）版本的交叉熵实现