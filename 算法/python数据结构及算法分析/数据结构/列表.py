# -- 无序列表：链表
# -- 节点（Node）

# -- 小乌龟 递归
# -- 子树绘制 递归

import turtle

def tree(branch_len, t):
    if branch_len > 5:
        t.forward(branch_len)
        t.right(20)
        tree(branch_len - 15, t)
        t.left(40)
        tree(branch_len - 15, t)
        t.right(20)
        t.backward(branch_len)

t = turtle.Turtle()
my_win = turtle.Screen()
t.left(90)
t.up()
t.backward(200)
t.down()
t.color("black")
tree(110, t)
my_win.exitonclick()

# -- 谢尔平斯基三角形
import turtle

def draw_triangle(points, color, my_turtle):
    my_turtle.fillcolor(color)
    my_turtle.up()
    my_turtle.goto(points[0][0], points[0][1])
    my_turtle.down()
    my_turtle.begin_fill()
    my_turtle.goto(points[1][0], points[1][1])
    my_turtle.goto(points[2][0], points[2][1])
    my_turtle.goto(points[0][0], points[0][1])
    my_turtle.end_fill()

def get_mid(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def sierpinski(points, degree, my_turtle):
    colormap = [
        "blue",
        "red",
        "green",
        "white",
        "yellow",
        "violet",
        "orange",
    ]
    draw_triangle(points, colormap[degree], my_turtle)
    if degree > 0:
        sierpinski(
            [
                points[0],
                get_mid(points[0], points[1]),
                get_mid(points[0], points[2]),
            ],
            degree - 1,
            my_turtle,
        )
        sierpinski(
            [
                points[1],
                get_mid(points[0], points[1]),
                get_mid(points[1], points[2]),
            ],
            degree - 1,
            my_turtle,
        )
        sierpinski(
            [
                points[2],
                get_mid(points[2], points[1]),
                get_mid(points[0], points[2]),
            ],
            degree - 1,
            my_turtle,
        )

def main():
    my_turtle = turtle.Turtle()
    my_win = turtle.Screen()
    my_points = [[-100, -50], [0, 100], [100, -50]]
    sierpinski(my_points, 5, my_turtle)
    my_win.exitonclick()

main()

# -- 汉洛塔问题
def move_tower(height, from_pole, to_pole, with_pole):
    if height < 1:
        return
    move_tower(height - 1, from_pole, with_pole, to_pole)
    move_disk(from_pole, to_pole)
    move_tower(height - 1, with_pole, to_pole, from_pole)

def move_disk(from_pole, to_pole):
    print(f"moving disk from {from_pole} to {to_pole}")

move_tower(3, "A", "C", "B")

# -- 快速排序
def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)

print(quicksort([10, 5, 2, 3]))



