"""
如果需要构建计算图，直接用张量进行计算
否则就需要取张量的数值进行计算
"""
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 创建权重
w = torch.tensor([1.0])
w.requires_grad = True  # 表示 w 是需要计算梯度的，在运行过程中会自动计算梯度的值

def forword(x):
    return x * w

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
        print(f'\tx = {x:.4f}, y = {y:.4f}, 损失函数对w的导数（权重） = {w.grad.item():.4f}')
        # 取 w.grad 的 data 进行计算，不会建立计算图，这里只是计算，不需要计算图
        w.data = w.data - 0.01 * w.grad.data
        # 权重里面的梯度数据全部清零，为下一次存储权重 w 做准备
        w.grad.data.zero_()
    print(f'\tEpoch = {epoch + 1}, 损失函数loss = {l.item():.6f}\n')

print(f"predict (after training 学习4小时可以得到 {forword(4).item():.2f} 分)")




