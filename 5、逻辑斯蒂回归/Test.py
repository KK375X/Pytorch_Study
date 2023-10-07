import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 0 表示未通过，1 表示通过
# 该数据集表示 1 小时未通过，2 小时未通过，3 小时通过
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# 设计模型
class LogisticRegressionModel(torch.nn.Module):
    # 构造函数，默认不用管
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # sigmoid() 函数求出预测值，实现二分类
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

# 创建对象
model = LogisticRegressionModel()

# 使用交叉熵求损失函数
criterion = torch.nn.BCELoss(reduction='sum')
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    # 求预测值
    y_pred = model(x_data)
    # 求损失函数
    loss = criterion(y_pred, y_data)
    print(f"Epoch = {epoch + 1},  loss = {loss.item():.5f}")

    # 清空梯度
    optimizer.zero_grad()
    # 反向传播求损失函数对权重的梯度
    loss.backward()
    # 更新权重
    optimizer.step()

# 绘图
# 对每周学习的时间 0 - 10 小时，采集 200 个点
x = np.linspace(0, 10, 200)
# 变成 200 行 1 列的矩阵
x_t = torch.Tensor(x).view(200, 1).to(torch.float32)
# 把测试张量送到模型里面，得到测试结果
y_t = model(x_t)
# 得到 n 维数组 y
y = y_t.data.numpy()
plt.plot(x, y)
# [0, 10] 表示线的长度，一般和 x 轴一样长
# [0.5, 0.5] 表示线出现的位置，与 y = 0.5 水平，两个数值相同表示水平，如果不相同就是斜线
# c='r' 表示 color = 'red'
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()

