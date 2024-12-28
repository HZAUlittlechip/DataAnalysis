# == 随机游走类的建立 ==
from random import choice

class RandomWalk:
    """ 生成随机游走的类 """
    def __init__(self,num_point=5000):
        """ 初始化随机游走的属性 """
        self.num_point = num_point

        # 游走的初始坐标
        self.x_value = [0]
        self.y_value = [0]

    def fill_walk(self):
        """ 计算随机游走的点 """

        # 循环判断何时停止游走
        while len(self.x_value) < self.num_point:

            #  随机生成需要 位移的距离
            x_direction = choice([1,-1]) # choic（[,]）
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance

            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance

            # 不原地踏步
            if x_step == 0 and y_step == 0:
                continue

            # 对比上个坐标 x_value[-1] 生成 下一个坐标
            x = self.x_value[-1] + x_step
            y = self.y_value[-1] + y_step

            # 添加进x,y的列表
            self.x_value.append(x)
            self.y_value.append(y)


