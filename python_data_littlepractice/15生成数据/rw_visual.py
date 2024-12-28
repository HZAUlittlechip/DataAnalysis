import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from random_walk import RandomWalk

# 循环生成随机游走的图
while True:
    # -- 创建 随机游走 的实例 --
    rw = RandomWalk()
    # rw = RandomWalk(50000)    # 增加游走点的个数的化可以在其中添加 实参
    rw.fill_walk()

    # -- 绘制出所有点 --
    plt.style.use('classic')
    # -- 修改输出图像的大小 --
    # 1. figsize=(,)是一个元组，返回单位为英寸
    # 2. dpi =     还可以添加dpi
    fig, ax = plt.subplots(figsize=(13.55, 8.47), dpi=189)

    # -- 根据 点的先后顺序来分配颜色 c & cmap --
    # 1. edgecolors = 'none' 去除点边缘的颜色
    # 2. set_aspect() 确保两轴的间距相等
    point_number = range(rw.num_point)
    ax.scatter(rw.x_value, rw.y_value, c=point_number, cmap=plt.cm.Blues,
               edgecolors='none',  s=15)
    ax.set_aspect('equal')

    # -- 单独突出点 或者可解释为 重新再绘制点 --
    ax.scatter(0,0, c='green', edgecolors='none', s=100)    # 初始点
    ax.scatter(rw.x_value[-1], rw.y_value[-1], c='red', edgecolors='none', s=100)

    # -- 隐藏坐标轴 --
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()

    keep_running = input('继续生成吗？继续’y‘,不继续’n‘')
    if keep_running == 'n':
        break