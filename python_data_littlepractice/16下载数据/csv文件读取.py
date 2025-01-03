from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from datetime import datetime

# -- 读取
# 1. 路径读取
path = Path(r'weather_data/sitka_weather_07-2021_simple.csv')
# 2. 逐行读取
lines =path.read_text().splitlines()

# -- csv 各行文件赋予
reader = csv.reader(lines)
# 1. 头行读取
head_row = next(reader)
print(head_row)
# 2. 头行索引标签读取
for index, column_header in enumerate(head_row):
    print(index, column_header)

# -- 读取数据的提取
# 1. 根据索引来读取最高温度
# 2. 提取日期且修改格式
# 3. 提取最低温度
highs, dates, lows = [], [], []
for row in reader:
    high = int(row[4])
    highs.append(high)
    low = int(row[5])
    lows.append(low)
    date = datetime.strptime(row[2], '%Y-%m-%d')
    dates.append(date)
print(highs)

# -- 温度图的绘制
# 1. 图像样式修改
print(plt.style.available)
plt.style.use('Solarize_Light2')
# 2. 画幅创建
fig, ax = subplots()
# 3. 图的创建(可以多图绘制)
ax.plot(dates, highs, color='red')
ax.plot(dates, lows, color='blue')
# 4. 标题和标签
ax.set_title("Daily High Temperatures, July 2021", fontsize=24)
ax.set_xlabel('',fontsize=16)
fig.autofmt_xdate()     #倾斜日期标签 防止重叠
ax.set_ylabel("Temperature (F)", fontsize=16)
ax.tick_params(labelsize=16)
# 5. 填充空的空间 -- fill_between()
ax.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

plt.show()

