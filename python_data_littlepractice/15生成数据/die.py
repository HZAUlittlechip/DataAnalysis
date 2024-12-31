from random import randint

class Die:
    """ 模拟投掷骰子 """
    def __init__(self, num_sides=6):
        """ 初始化骰子初始的面数 """
        self.num_side = num_sides

    def roll(self):
        return randint(1,self.num_side)