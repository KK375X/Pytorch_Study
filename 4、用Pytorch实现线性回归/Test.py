import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 编写学习模型模块
# 所有的模型都要继承自 torch.nn.Module 模块
class LinearModel(torch.nn.Module):
    # 构造函数 __int__() 和 forward() 必须要实现
    def __init__(self):
        # 调用父类的构造函数，不用管，直接写就行，必须要有
        super(LinearModel, self).__init__()
        # 构造对象，包含了权重 w 和偏置项 b
        # in_features：定义输入样本的维数
        # out_features：定义输出样本的维数
        # bias：是否需要偏置项 b，默认为 true（要）
        self.linear = torch.nn.Linear(1, 1)

    # 前馈计算，返回预测值
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    # 由于 Module 模块可以根据计算图自动计算 backward() 的反馈部分，所以不需要写 backward() 函数求梯度

# 实例化，创建 model 对象，这个对象就是训练的模型，后期将测试数据丢入这个对象中即可
model = LinearModel()

# 构造损失函数和优化器
# size_average=False 已经被新版本弃用，使用 reduction='sum' 可以达到相同的效果
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练过程
for epoch in range(100):
    # 前馈计算
    # 利用前面定义的 model 对象得到预测值
    y_pred = model(x_data)
    # 得到损失函数 loss，criterion 是损失函数的实例化对象
    loss = criterion(y_pred, y_data)
    print(f'Epoch = {epoch + 1},   loss = {loss:.5f}')

    # 告诉优化器将梯度清零
    optimizer.zero_grad()
    # 3、反向传播，得到梯度
    loss.backward()
    # 更新权重
    optimizer.step()

# 输出权重 w 和偏置项 b
print(f'w = {model.linear.weight.item():.5f},  b = {model.linear.bias.item():.5f}')

# 通过训练集得到模型之后，需要用测试集进行测试
# 输入测试集数据
x_test = torch.Tensor([[4.0]])
# 将测试集数据传入模型中，得到预测值
y_test = model(x_test)

# 打印预测值
print(f'学习 4 小时，预计可以获得 {y_test.item():.2f} 分')



