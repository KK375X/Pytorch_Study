import numpy as np
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 线性模型
def forward(x, w, b):
    return x * w + b


# 损失函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) * (y_pred - y)


def mse(w, b):
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val, w, b)
        loss_val = loss(x_val, y_val, w, b)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    return l_sum / 3


# 迭代取值，计算每个w取值下的x，y，y_pred,loss_val
mse_list = []

##画图

##定义网格化数据
b_list = np.arange(-30, 30, 0.1)
w_list = np.arange(-30, 30, 0.1);

##生成网格化数据
xx, yy = np.meshgrid(b_list, w_list, sparse=False, indexing='xy')

##每个点的对应高度
zz = mse(xx, yy)

fig = plt.figure()
ax = Axes3D(fig)
plt.show()
