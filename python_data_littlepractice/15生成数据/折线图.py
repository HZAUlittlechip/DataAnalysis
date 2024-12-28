import matplotlib.pyplot as plt

squares = [1, 4, 9, 16, 25]
input_values = [1, 2, 3, 4, 5]

# 使⽤内置样式
# == 展示样式 ==
print(plt.style.available)
plt.style.use('Solarize_Light2')

# == ax ---> Axes ==
fix, ax = plt.subplots()

#  修改标签⽂字和线条粗细
# == 调整线宽 ==
# == plot（x，y） ==
ax.plot(input_values,squares,linewidth = 3)  # 都是对ax来进行的操作

# == 设计图题且加上label ==
# == fontsize 用来表示文字的大小 ==
ax.set_title("Square Numbers", fontsize = 24)
ax.set_xlabel("Value", fontsize = 14)
ax.set_ylabel("Square of Value", fontsize = 14)

# == 设置刻度标记的样式 ==
ax.tick_params(labelsize = 14)

plt.show()