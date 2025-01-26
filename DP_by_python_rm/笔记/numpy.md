# nupmy

## 常用函数

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
print(x2) # [51 55 19]
```

- `.arange()`: 按步长生成数组

```python
x = np.arange(0, 6, 0.1) # 0-6 步长为0.1的数组
```

- `np.ndim()`:查看数组的维度

```python
x1 = np.array([[51, 55], [14, 19], [0, 4]])
np.ndim(x1) # 2
```

- `.shape`:查看数组的形状

```python
x1 = np.array([[51, 55], [14, 19], [0, 4]])
x1.shape # (3,2)
x1.shape[0] # 3 第一个维度是0维
x1.shape[1] # 2
```

- `np.dot(a, b)`:矩阵的乘法

```python
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
np.dot(A, B) 
# array([[19, 22],
#      [43, 50]])
```

- `np.sum(a, axis= )`:矩阵的求和

1. 对整个矩阵求和

    ```python
    import numpy as np
    a = np.array([[1, 2], 
                  [3, 4]])
    print(np.sum(a)) # 10 
    ```

2. 沿特定的轴求和

```python
# 沿列的方向来求和 axis=0
print(np.sum(a, axis=0)) # [3, 7]  [1+2, 3+4]

# 沿行的方向来求和 axis=1
print(np.sum(a, axis=1)) # [4, 6]  [1+3, 2+4]
```

