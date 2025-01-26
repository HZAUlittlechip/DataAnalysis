# matplotlib

## image包

- `imread(r'')`：导入图片
- `plt.imshow`: 读入图片到plt类中

```python
improt matplotlib.pyplot as plt
form matplotlib.image improt imread

img = imread('pic.png') 
plt.imshow(img)

plt.show()
```