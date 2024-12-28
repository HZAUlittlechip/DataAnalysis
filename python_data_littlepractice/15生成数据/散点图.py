# 散点图
# == scatter() ==
import matplotlib.pyplot as plt
import numpy as np

print(plt.style.available)
plt.style.use('_classic_test_patch')

fig, ax = plt.subplots()

# == 自己输入数据 ==
# x_values = [1, 2, 3, 4, 5]
# y_values = [1, 4, 9, 16, 25]

# == 自动生成数据 ==
x_values = range(1,1001)
y_values = [x**2 for x in x_values]

# == color 红 绿 蓝  s -- 修改点的大小 ==
# ax.scatter(x_values,y_values,s= 10,color=(0,0.8,0))
# == cmap=plt.cm.Blues 蓝色渐变色 ==
ax.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues, s=10)

# == 设置坐标轴取值范围 和 刻度样式 ==
ax.axis([0, 1100, 0, 1_100_000])
ax.ticklabel_format(style='scientific')  # 以科学技术发的形式显示数字

# == 标签修改 ==
ax.set_title("Square Numbers", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)

# == 修改坐标轴标签刻度外观 ==
ax.tick_params(labelsize=14)


# == 保存图像 ==
# 自动保存一定要在show()前面实现，不然会生成空白图像
plt.savefig('squares_plot.png', bbox_inches='tight')
# == 自定义路径保存 ==
# plt.savefig('squares_plot.png', bbox_inches='tight',path='')
# == matplotlib 查看器显示图片 ==
plt.show()






# == subplots == 多图共舞
x = np.linspace(0,2*np.pi,400)
y = np.sin(x**2)
f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.plot(x,y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x,y)

plt.show()