# 2、梯度下降算法
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 权重初始猜测，先定义为 1.0
w = 1.0

# 定义训练次数和损失函数列表
epoch_list = []
loss_list = []

# 前馈计算
def forword(x):
    return x * w

# 计算均方差 MSE
def cost(xs, ys):
    cost_sum = 0
    for x, y in zip(xs, ys):
        y_pred = forword(x)
        cost_sum += (y_pred - y) ** 2
    return cost_sum / len(xs)


# 计算梯度
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        # 梯度求和
        grad += 2 * x * (x * w - y)
    return grad / len(xs)  # 返回梯度的平均值

# round(forword(4), 2) 对要输出的 forword(4) 保留两位小数
print(f'Predict (before training 学习4小时能得到{round(forword(4), 2)}分)')

for epoch in range(100):
    # 均方差 mse
    cost_val = cost(x_data, y_data)
    # 梯度
    grad_val = gradient(x_data, y_data)
    # 将训练次数和损失函数存储起来，方便作图
    epoch_list.append(epoch)
    loss_list.append(cost_val)
    # 0.01 为学习率 α
    w = w - 0.01 * grad_val
    print(f'Epoch = {epoch + 1}, 权重w = {w:.2f}, 损失函数loss = {cost_val:.2f}, 梯度grad = {grad_val:.2f}')
print(f'Predict (after training 学习4小时能得到{round(forword(4), 2)}分)')

# 绘图
plt.plot(epoch_list, loss_list)
plt.xlabel(epoch_list)
plt.ylabel(loss_list)
plt.show()
