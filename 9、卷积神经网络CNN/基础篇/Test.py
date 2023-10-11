import torch
from torchvision import transforms  # 对图像进行处理的库
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用 relu() 作为激活函数
import torch.optim as optim  # 优化器的库

"""
图像格式转换
"""
batch_size = 64

transform = transforms.Compose([
    # 使用 ToTensor() 将图像格式转换成 Pytorch 能处理的的多通道张量格式
    transforms.ToTensor(),
    # 归一化，Normalize(mean:均值, std:标准差) 整个样本数据里面算出来的经验值
    transforms.Normalize((0.1307,), (0.3081,))
])

"""
加载数据集
自己的电脑：D:/MyProjects/Pytorch/Dataset/MNIST/
工位：D:/MyProject/Pytorch/Dataset/MNIST/
"""
train_dataset = datasets.MNIST(root='D:/MyProject/Pytorch/Dataset/MNIST/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='D:/MyProject/Pytorch/Dataset/MNIST/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

"""
模型设计
"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # 第二个卷积层
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 池化层（由于没有设置权重，只用一个池化层就行）
        self.pooling = torch.nn.MaxPool2d(2)
        # 全连接层为 320 映射为 10 的线性层
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        # 从 size 中拿出数据的维度 n
        batch_size = x.size(0)
        # 从内向外的顺序是先做卷积，再做池化，最后做 relu 激活
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # 转变 x 的形式，变成全连接网络需要的输入形式
        x = x.view(batch_size, -1)  # flatten
        # 最后求交叉熵损失，不用做激活
        x = self.fc(x)
        return x

model = Net()
"""
使用 CUDA 显卡加速在此处添加以下语句：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
"""
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# 随机梯度下降 SGD 优化器，momentum=0.5 表示带冲量的优化器，优化训练过程
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

"""
训练与测试
"""
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # inputs 作为输入，target 作为输出
        inputs, target = data
        """
        使用 CUDA 显卡加速在此处添加以下语句：
        inputs, target = inputs.to(device), target.to(device)
        """
        # 梯度清零
        optimizer.zero_grad()
        # forward 前馈
        outputs = model(inputs)
        loss = criterion(outputs, target)
        # backward 反馈
        loss.backward()
        # update 更新
        optimizer.step()

        # 累计 loss 的数值
        running_loss += loss.item()

        # 每300轮迭代后输出 Loss
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    # 正确的有多少
    correct = 0
    # 总数有多少
    total = 0
    # test不需要计算梯度，因此使用 torch.no_grad() 可以提升速度
    with torch.no_grad():
        # 过程：从 test_loader 里面取得数据，放入 model 模型中做预测
        for data in test_loader:
            # 取数据
            images, labels = data
            # 将图像数据放入 model 模型中
            outputs = model(images)
            # 沿着 dim=1（行） 寻找最大值的下标，返回两个值：第一个：最大值是多少；第二个：最大值的下标是多少
            _, predicted = torch.max(outputs.data, dim=1)  # torch.max返回值有两个，最大值的下标+最大值的大小；
            # 例如，一个图像是 (N, 1)，则表示加上 N
            total += labels.size(0)
            # 将 predicated 预测值与 labels 正确值做比较，预测对了就 +1，错误不加
            correct += (predicted == labels).sum().item()
    print(f'正确率为: {(100 * correct / total)}%')


if __name__ == "__main__":
    for epoch in range(5):
        train(epoch)
        test()
