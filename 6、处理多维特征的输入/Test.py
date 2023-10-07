import torch.nn
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
# delimiter=',' ：分隔符
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
# (xy[:, :-1]) 表示拿出所有数据，最有一列不要，最后一列表示的是结果，也就是糖尿病人是否得病，这个需要后面计算出来
x_data = torch.from_numpy(xy[:, :-1])
# (xy[:, [-1]]) 表示拿出最后一列，[-1] 表示将最后一列形成一个矩阵
y_data = torch.from_numpy(xy[:, [-1]])

# 设计模型
class Model(torch.nn.Module):
    # 构造函数，默认不用管
    def __init__(self):
        super(Model, self).__init__()
        # 输入维度：8   输出维度：6，表示可以把 8 维空间的矩阵映射到 6 维空间上，Linear() 可以做到空间维度的变换
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # torch.nn.Sigmoid() 就是添加非线性的变换，也就是 1 / (1 + e^(-z))，这里是一个模块，与下面的 self.sigmoid() 函数不一样
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # self.sigmoid() 函数求出预测值，实现二分类
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

# 创建对象
model = Model()

# 创建绘图用的坐标
x_list = []     # 用于记录循环次数
y_list = []     # 用于记录损失函数

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
    print(f"Epoch = {epoch},  loss = {loss.item():.5f}")

    x_list.append(epoch)
    # int(loss.item()) 使 loss 展示的是整数
    y_list.append(int(loss.item()))

    # 清空梯度
    optimizer.zero_grad()
    # 反向传播求损失函数对权重的梯度
    loss.backward()
    # 更新权重
    optimizer.step()

# 绘图
plt.plot(x_list, y_list)
plt.xlabel(x_list)
plt.ylabel(y_list)
plt.show()

