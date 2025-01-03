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

# 15.4 改进随机游走
# ** 改进方向 **
# 1.修改成更远的距离
# 2.去掉随机游走的方向
rw = RandomWalk()
rw.fill_walk()


plt.style.use('classic')

fig, ax = plt.subplots(figsize=(13.55, 8.47), dpi=189)


point_number = range(rw.num_point)
ax.scatter(rw.x_value, rw.y_value, c=point_number, cmap=plt.cm.Blues,
           edgecolors='none',  s=15)
ax.set_aspect('equal')


ax.scatter(0,0, c='green', edgecolors='none', s=100)    # 初始点
ax.scatter(rw.x_value[-1], rw.y_value[-1], c='red', edgecolors='none', s=100)

# -- 隐藏坐标轴 --
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

# 15.6 两个D8
from die import Die
import plotly.express as px

die1 = Die(8)
die2 = Die(8)

# -- 记入结果
results = []
for i in range(1000):
    result = die1.roll() + die2.roll()
    results.append(result)

# -- 分析结果
frequencies = []
chance_die = range(2,die1.num_side + die2.num_side + 1)
for i in chance_die:
    frequency = results.count(i)
    frequencies.append(frequency)

# -- 结果出图
title = '八面骰子投掷1000次的频率直方图'
labels = {'x':'投掷结果','y':'投掷频次'}
fig = px.bar(x=chance_die, y=frequencies, title=title, labels=labels)
fig.show()

# -- 15.7 同时投掷三个骰子
from die import Die
import plotly.express as px

die1 = Die()
die2 = Die()
die3 = Die()

# -- 记入结果
results = []
for i in range(1000):
    result = die1.roll() + die2.roll() + die3.roll()
    results.append(result)

# -- 分析结果
frequencies = []
chance_die = range(3,die1.num_side + die2.num_side + die2.num_side+ 1)
for i in chance_die:
    frequency = results.count(i)
    frequencies.append(frequency)

# -- 结果出图
title = '三面D6骰子投掷1000次的频率直方图'
labels = {'x':'投掷结果','y':'投掷频次'}
fig = px.bar(x=chance_die, y=frequencies, title=title, labels=labels)
# - 显示全部的x轴标签
fig.update_layout(xaxis=dict(tickmode='linear', tickangle=0))
fig.show()

# --15.8 easy

# -- 15.9 列表推导式子的修改
# ①
# results = []
# for i in range(1000):
#     result = die1.roll() + die2.roll() + die3.roll()
#     results.append(result)
results = [die1.roll() + die2.roll() + die3.roll() for _ in range(1000)]

# ②
# frequencies = []
# chance_die = range(3,die1.num_side + die2.num_side + die2.num_side+ 1)
# for i in chance_die:
#     frequency = results.count(i)
#     frequencies.append(frequency)
frequencies = [results.count(value) for value in chance_die]

# -- 15.10 随机游走的plotly实现 和 matplotlib实现随机骰子
# ① 随机游走
from random_walk import RandomWalk
import plotly.express as px

rw = RandomWalk()
rw.fill_walk()

# --绘图
titles = 'plotly的随机游走'
# - 渐变功能的实现
# 1.以点出现的索引来构建颜色
point_index = list(range(rw.num_point))
fig = px.scatter(x=rw.x_value, y=rw.y_value,title=titles,color=point_index)
fig.show()
