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

            x_step = self.get_step()
            y_step = self.get_step()

            # 不原地踏步
            if x_step == 0 and y_step == 0:
                continue

            # 对比上个坐标 x_value[-1] 生成 下一个坐标
            x = self.x_value[-1] + x_step
            y = self.y_value[-1] + y_step

            # 添加进x,y的列表
            self.x_value.append(x)
            self.y_value.append(y)

    def get_step(self):
        """ 确定游走的步长 """
        direction = choice([1, -1])
        distance = choice([0, 1, 2, 3, 4, 5])
        step = distance * direction
        return step



