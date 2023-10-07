"""
练习题
    已知 y_pred = (w1 * x ** 2) + (w2 * x) + b
    计算损失函数 loss = (y_pred - y) ** 2 对 w1，w2 和 b 的导数
"""
import torch
import math

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 创建权重
w1 = torch.tensor([1.0])
w2 = torch.tensor([1.0])
b = torch.tensor([1.0])
w1.requires_grad = True  # 表示 w 是需要计算梯度的，在运行过程中会自动计算梯度的值
w2.requires_grad = True
b.requires_grad = True

def forword(x):
    return (w1 * math.pow(x, 2)) + (w2 * x) + b

def loss(x, y):
    y_pred = forword(x)
    return (y_pred - y) ** 2

print(f"predict (before training 学习4小时可以得到 {forword(4).item()} 分)")

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 计算损失函数 loss，是一个张量
        l = loss(x, y)
        # 把整个计算链路上需要求梯度的地方全部求出梯度，并存储到 w 里面，因为 w 在前面已经设置过 w.requires_grad = True
        # 一旦调用了 backward()，就把之前的计算图全部清楚了，下次循环重新建立起新的计算图
        l.backward()
        print(f'\tx = {x}, y = {y}, 对w1的导数 = {w1.grad.item():.4f}, 对w2的导数 = {w2.grad.item():.4f}, 对b的导数 = {b.grad.item():.4f}')
        # 取 w.grad 的 data 进行计算，不会建立计算图，这里只是计算，不需要计算图
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        # 权重里面的梯度数据全部清零，为下一次存储权重 w 做准备
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    # l.item() 只取值，不带张量的属性
    print(f'\tEpoch = {epoch + 1}, 损失函数loss = {l.item():.6f}\n')

print(f"predict (after training 学习4小时可以得到 {forword(4).item():.2f} 分)")
