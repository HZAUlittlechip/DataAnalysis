# --16.1 锡特卡的降⾬量
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

# -- 读取数据
path = Path(r'weather_data/sitka_weather_2021_full.csv')
lines = path.read_text().splitlines()

# -- 存为csv格式
reader = csv.reader(lines)
# 1. 标签读取
header_row = next(reader)
for index, column in enumerate(header_row):
    print(index, column)        # 5 PRCP ; 2 DATE

# -- 读取数据保存
rains, dates  = [], []
for row in reader:  # 已经读取过一次 next(reader) 不会显示第一行了
    date = datetime.strptime(row[2],'%Y-%m-%d')
    dates.append(date)
    rain = float(row[5])
    rains.append(rain)

# -- 绘图
print(plt.style.available)
plt.style.use('seaborn')
fig, ax=subplots()

ax.plot(dates, rains, color='blue')

# -- 

plt.show()


