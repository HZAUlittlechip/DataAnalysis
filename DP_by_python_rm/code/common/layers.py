import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x , y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout=1):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout=1):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        out = x.copy
        self.mask = (x <= 0)
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid():
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        return out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out

        return dx