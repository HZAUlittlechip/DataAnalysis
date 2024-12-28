import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import lineStyles
from scipy.interpolate import make_interp_spline
from matplotlib.collections import LineCollection

from random_walk import RandomWalk

print(plt.style.available)

# == 设置中文字体 ==
plt.rcParams['font.sans-serif'] = ['simHei']  # 使用黑体(全局修改)
plt.rcParams['axes.unicode_minus'] = False    # 防止负号显示为方块

# 15.1 & 15.2 立方
fig, ax = plt.subplots()

x_value = range(1,5001)
y_value = [x**3 for x in x_value]

# == 彩色映射 ==
# c=y_value 的意思是根据 y_value 的值来分配颜色
# 单独的颜色的话直接用 color( , , ) 就行
ax.scatter(x_value,y_value,c=y_value,cmap=plt.cm.inferno,s=10)

# 添加标签
ax.set_title("立方数", fontsize=18)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("x^3", fontsize=14)

# 修改坐标轴的刻度
ax.tick_params(labelsize=14)  # lablesize 不是 fontsize

plt.show()

# 15.3 分子运动
# *** 改进方案 ***
# 1. 添加颜色渐变 -- ax.plot()本身不支持颜色渐变 ，
#    但可以分段绘制路径且通过 cmap来模拟渐变
# 3. 平滑曲线 --- scipy.interpolate
# 4. 高质量显示 --- linewidth 和 dpi
# —————————————————————————————————————————————————————————
# -- plot 函数 --
rw = RandomWalk(5000)
rw.fill_walk()

# -- 绘制出所有点 --
plt.style.use('seaborn-bright')

# -- 修改输出图像的大小 --
fig, ax = plt.subplots(figsize=(13.55, 8.47), dpi=189)

# -- 去除重复的 x值，并保持y值同步
unique_x, unique_indices = np.unique(rw.x_value, return_index=True)
unique_y = np.array(rw.y_value)[unique_indices]

# -- 描述不同点出现的顺序 --
# 1. plot 不是散点图，c 和 cmap 是无法生效的 ！
point_number = range(rw.num_point)
norm = Normalize(vmin=0, vmax=rw.num_point)  # 归一化点数的范围
cmap = plt.cm.Blues     # 蓝色渐变

# -- 循环为每段添加颜色渐变 --
for i in range(1, rw.num_point):
    x_values = rw.x_value[i-1:i+1]
    y_values = rw.y_value[i-1:i+1]
    ax.plot(x_values, y_values,color=cmap(norm(i)),linewidth=2,alpha=0.75)

# -- 单独突出点 或者可解释为 重新再绘制点 --
ax.scatter(0, 0, c='green', edgecolors='none', s=100)  # 初始点
ax.scatter(rw.x_value[-1], rw.y_value[-1], c='red', edgecolors='none', s=100)

# -- 隐藏坐标轴 --
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# -- 设置图像比例 --
ax.set_aspect('equal')  # set_aspect() 确保两轴的间距相等
ax.margins(0.1)     # 设置xy的边轴距离

plt.show()