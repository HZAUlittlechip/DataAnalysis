import pickle
import sys, os
from DP_by_python_rm.code.common.functions import *
sys.path.append(os.pardir)
''' dataset.mnist 是自己写的快速导入MNIST的包 '''
from PIL import Image
from dataset.mnist import load_mnist
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False) # 展开为1维数组，且讲像素保留为原来的0 ~ 255

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# 图像展示
img = x_train[0]
label = t_train[0]
print(label) # 5
print(img.shape) # (784,)
img = img.reshape(28, 28) # 修改图像为原始尺寸
print(img.shape) # (28, 28)
img_show(img)

def get_data():
    """ 输出测试数据集 """
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 注意需要正则化，否则sigmoid函数会计算溢出

    return x_test, t_test

def init_network():
    """ 存储权重和偏置 """
    with open("sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)  
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# 精度识别判断
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print(f"Accuracy:{str(float(accuracy_cnt) / len(x))}")

x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

batch_size = 100
accuracy_cnt = 0

# 批处理
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 沿第一维轴来进行最大的元素寻找
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print(f"Accuracy:{str(float(accuracy_cnt) / len(x))}")



