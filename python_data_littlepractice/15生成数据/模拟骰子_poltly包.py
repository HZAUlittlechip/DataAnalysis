"""
            plotly 函数交互性绘图的包
1. 其为 pandas下面的包
2. plotly包的官网 ：‘https://plotly.com/python/’
"""
from die import Die
import plotly.express as px

# -- 骰骰子 --
# - 创建实例
die = Die()
# - 记入生成的点数且将其存入列表中
results = []
for roll_num in range(1000):
    result = die.roll()
    results.append(result)
print(results)

# -- 分析结果 --
# - 读取每个点数的出现的频率
frequencies = []
# - 列举骰子可能出现的情况
poss_results = range(1,die.num_side+1)
# - 依次检查骰出的结果
for value in poss_results:
    frequency = results.count(value)
    frequencies.append(frequency)
print(frequencies)

# -- 绘制直方图 --
# - px.bar 绘制直方图
fig = px.bar(x=poss_results, y=frequencies)
fig.show()

# -- 定制绘图 --
# - 添加标签
# 1. 坐标轴标签可以用字典的形式生成
title = '骰100次骰子的频率直方图'
labels ={'x':'投掷结果','y':'出现频次'}
fig = px.bar(x=poss_results, y=frequencies, title=title, labels=labels)
# 2. 显示完整的x轴标签
fig.update_layout(xaxis=dict(tickmode='linear', tickangle=0))
fig.show()

# -- html 格式图像的保存
# - json , image 格式也是可以的
# 1. 函数内填入文件的名称 like 'dice_visual_d6d10.html'
fig.write_html('dice_visual_d6d10.html')