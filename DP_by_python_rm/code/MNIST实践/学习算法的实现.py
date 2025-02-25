import numpy as np
import os, sys
sys.path.append(os.pardir)
from DP_by_python_rm.code.common.functions import *
from dataset.mnist import load_mnist

class TwoLayerNet:
    """ """
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = { }
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        y = softmax(z2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """ 较为低效后续会替换
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])

        return grads

# mini-batch和测试数据评价实例演示
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)


# 超参数设置
iters_num = 10000 # 梯度法的循环次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini——batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度计算
    grad = network.numerical_gradient(x_batch, t_batch) \
        # 这个梯度计算的方法确实巨慢

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc |  {str(train_acc)}, {str(test_acc)}")








