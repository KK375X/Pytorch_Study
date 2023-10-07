import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""
准备数据集
"""
# DiabetesDataset(Dataset) 表示 DiabetesDataset 类继承了 Dataset 父类的功能
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # xy.shape[0] 由于 shape 为 (N, 9)，因此需要将有多少行取出来，也就是第一个参数 N
        self.len = xy.shape[0]
        # 由于数据集很小，直接把他们都存储在内存中，也就是 self 中
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # __getitem__(self, index) 使数据集中的样本具有下标，可以通过下标找到样本位置
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 创建处理数据集的对象，并把数据集作为参数传递进去
dataset = DiabetesDataset('diabetes.csv.gz')

"""
DataLoader() 数据加载器，对数据进行操作
    - dataset=dataset 传递数据集，指明需要加载哪里的数据
    - batch_size=2 一个小批量的数据集中有多少个数据样本，也就是一个 min-batch
    - shuffle=True 是否打乱原始数据集中的数据样本
    - num_workers=2 在都样本的时候是否要用多线程，即需要几个并行的进程读取数据
"""
train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=2)

"""
设计模型
"""
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

"""
构造损失函数和优化器
"""
# 使用交叉熵求损失函数
criterion = torch.nn.BCELoss(reduction='sum')
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""
训练周期
"""
if __name__ == '__main__':
    # 外层循环对 min-batch 的数量进行迭代
    for epoch in range(10):
        # 内层循环对 min-batch 中的样本进行迭代，参数中的 0 表示每次都从 min-batch 数据集中下表为 0 的位置进行迭代
        for i, data in enumerate(train_loader, 0):
            # 将需要输入的 inputs 和标签 labels 从 data 数据集中取出来
            inputs, labels = data

            # 前馈计算
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(f'epoch = {epoch}, i = {i}, loss.item() = {loss.item():.3f}')

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新权重
            optimizer.step()

        print("------------------------------------------------------------------")
