from pathlib import Path
import json
import plotly.express as px
import pandas as pd

# -- 以 ‘地震数据为例’
# -- 将数据以字符串的形式读取 后 存为 python格式
path = Path(r'eq_data/eq_data_1_day_m1.geojson')
contents = path.read_text()
all_eq_data = json.loads(contents)

# -- 便于json格式数据阅读再次保存
path = Path(r'eq_data/eq_data_1_day_canread.geojson')
# 1. 重排数据 indent = 4  增加缩进
readable_contents = json.dumps(all_eq_data, indent=4)
# 2. 写入重排数据以字符串的形式
path.write_text(readable_contents)

# -- 有用的 键 有 features, coordinates, properties

# -- 读取所以地震的要素
all_eq_dict = all_eq_data['features']
print(len(all_eq_dict))     # 160

# -- 读取地震中的震级 properties --> mag
mags = []
for eq_dict in all_eq_dict:
    mag = eq_dict['properties']['mag']
    mags.append(mag)

# -- 读取 标题 经纬度(lons, lats) 内容
titles, lons, lats = [], [], []
for eq_dict in all_eq_dict:
    title = eq_dict['properties']['title']
    lon = eq_dict['geometry']['coordinates'][0]
    lat = eq_dict['geometry']['coordinates'][1]
    titles.append(title)
    lons.append(lon)
    lats.append(lat)

print(mags[:10])
print(titles[:2])
print(lons[:5])
print(lats[:5])

# -- 散点图绘制(plotly)
fig = px.scatter(x=lons, y=lats, labels={'x':'经度', 'y':'纬度'},
                 range_x=[-200, 200], range_y=[-90,90],
                title='全球地政散点图')

# 1. 图像保存
fig.write_html('global_earthquakes.html')

# 2. pandas 数据的封装 可以简洁 scatter中的代码
data = pd.DataFrame(data=zip(lons, lats, titles,mags),
                    columns=['经度', '纬度', '位置', '震级'])
# - 生成为表格型式的内容 160 row * 4 columns
print(data)

# 3. 标记定制
# - 根据 mag的大小来修改点的大小
fig = px.scatter(data, x='经度',y='纬度',
                 range_x=[-200, 200], range_y=[-90,90],
                title='全球地政散点图',
                 size='震级', size_max=10,
                 color='震级',
                 hover_name='位置')
fig.show()


