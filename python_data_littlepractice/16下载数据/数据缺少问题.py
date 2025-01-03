import csv
from datetime import datetime
from pathlib import Path

# -- 读取数据
path = Path(r'weather_data/death_valley_2021_simple.csv')
lines = path.read_text().splitlines()

# --csv 存储
read = csv.reader(lines)
headed_row = next(read)

# -- 读取索引
for index, label in enumerate(headed_row):
    print(index, label)

# -- 提取数据
# 1.因为 数据的缺少 会报错
# 2. 忽略缺少数据 用 try - expect - else
dates, highs, lows = [], [], []
for row in read:
    current_date = datetime.strptime(row[2], '%Y-%m-%d')
    try:
        high = int(row[3])
        low = int(row[4])
    except ValueError:
        print(f'Missing data for {current_date}')
    else:
        dates.append(current_date)
        highs.append(high)
        lows.append(lows)


