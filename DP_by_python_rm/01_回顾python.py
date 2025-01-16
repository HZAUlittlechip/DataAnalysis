""" nupmy & matplotlib """
import numpy as np
import matplotlib.pyplot as plt



# -- np数组
x = np.array([1.0, 2.0, 3.0])

print(x)
print(type(x))

# -- np的算数运算
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(x + y) # [3. 6. 9.]
print(x - y) # [-1. -2. -3.]
print(x / y) # [0.5 0.5 0.5]
print(x * y) # [ 2.  8. 18.]
print(x / 2.0) # [0.5 1.  1.5]

# -- np的多维数组
# 1. 二维数组
a = np.array([[1, 2], [3, 4]])
print(a)
# .shape 查看矩阵的尺寸 & .dtype 查看矩阵元素的数据类型
print(a.shape) # (2, 2)
print(a.dtype) # int32
# 2. 矩阵间的运算
b = np.array([[3, 0], [0, 6]])
print(a + b) #[[ 4, 2],
             # [ 3, 10]]
print(a * b) # [[ 3, 0],
             # [ 0, 24]]

# -- np中的广播
# 不同尺寸的数组间也能进行运算，np会调整或者说是扩充小矩阵变成和大矩阵同尺寸，再进行运算

# -- 访问元素
x1 = np.array([[51, 55], [14, 19], [0, 4]]) # 3 * 2的矩阵,其实是三个元素
print(x1)
print(x1[0]) # [51 55]
print(x1[0][1]) # 55

# - 遍历全部元素
# 1.以列的形式输出
for row in x1:
    print(row)  # [51 55]；[14 19]；[0 4]
# 2.输出每个元素
# 2，1 .flatten() 将矩阵转为一维数组
x1 = x1.flatten()
print(x1) # [51 55 14 19  0  4]
# 2.2 以数组形式输出
x2 = x1[np.array([0, 2, 4])]
print(x2) # [51 14  0]
# 2.3 输出矩阵中大于15的值
print(x1 > 15) # [ True  True False  True False False]
x3 =x1[x1>15]
print(x3)  # [51 55 19]




# -- matplotlib




