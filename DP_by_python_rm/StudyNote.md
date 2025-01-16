# 深度学习入门——基于python

## nupmy

### np库常用函数

- `np.array`：创建矩阵

```python
import nupmy as np
x1 = np.array([[51, 55], [14, 19], [0, 4]])
print(x1)
```

- `.flatten()`:将矩阵压缩一维数组

```python
x1 = np.array([[51, 55], [14, 19], [0, 4]])
x1 = x1.flatten()
print(x1) # [51 55 14 19  0  4]
```

- `array[array > value]`:输出矩阵中大于某值的对应元素

```python
x1 = np.array([[51, 55], [14, 19], [0, 4]])
x2 = x1[x1 > 15]
print(x2) #  # [51 55 19]
```



