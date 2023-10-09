import torch
from torchvision import transforms  # 对图像进行处理的库
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用 relu() 作为激活函数
import torch.optim as optim  # 优化器的库

batch_size = 64

transform = transforms.Compose([
    # 使用 ToTensor() 将图像格式转换成 Pytorch 能处理的的多通道张量格式
    transforms.ToTensor(),
    # 归一化，Normalize(mean:均值, std:标准差) 整个样本数据里面算出来的经验值
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
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
