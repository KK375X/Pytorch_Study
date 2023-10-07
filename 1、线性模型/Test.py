# 作图库
import numpy as np
import matplotlib.pyplot as plt

# 数据集，x 为输入，y 为输出
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 定义模型 X * W
def forword(x):
    return x * w

# 定义损失函数
def loss(x, y):
    y_pred = forword(x)
    return (y_pred - y) * (y_pred - y)

# 保存权重 W 和 损失值 MSE
w_list = []
mse_list = []

# w 从 0.0 穷举到 4.0，步长为 0.1
for w in np.arange(0.0, 4.1, 0.2):
    # 定义损失的和，方便后面求均方差
    l_sum = 0
    # 将 x_data 和 y_data 拿出来，用 zip 拼成 x 和 y 的数据值
    for x_val, y_val in zip(x_data, y_data):
        # 计算预测值
        y_pred_val = forword(x_val)
        # 计算损失
        loss_val = loss(x_val, y_val)
        # 对损失进行求和
        l_sum += loss_val
        # 输出 x，y，y 的预测值和损失函数 loss 的值
        print(f"\tx_val={x_val:.4f}, y_val={y_val:.4f}, y_pred_val={y_pred_val:.4f}, loss_val={loss_val:.4f}, l_sum={l_sum:.4f}")
    print(f"权重w = {w:.4f}, 均方差MSE = {(l_sum / 3):.4f}")
    print("------------------------------------------------")
    # 记录权重 w 和 均方差 mse 的值，对应位置相匹配，为作图做准备
    w_list.append(w)
    mse_list.append(l_sum / 3)

# 绘图
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

