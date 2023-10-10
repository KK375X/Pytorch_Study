import torch

# 输入通道数，输出通道数
in_channels, out_channels = 5, 10
# 图像的大小
width, height = 100, 100
# 卷积核的大小，参数为 3 表示 3*3
kernel_size = 3
# 表示小批量喂入数据
batch_size = 1

input = torch.randn(batch_size, in_channels, width, height)

# 创建卷积对象，卷积层模块 Conv2d，后面三个参数必须要写
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

# 将数据喂入卷积层对象
output = conv_layer(input)

"""
input.shape = torch.Size([1, 5, 100, 100]) ：输入数据中通道数为5，大小为100*100
output.shape = torch.Size([1, 10, 98, 98]) ：输出数据中通道数为10，大小为98*98
conv_layer.weight.shape = torch.Size([10, 5, 3, 3]) ：卷积层中输出的通道为10，输入的通道为5，卷积核的大小为3*3
"""
print(f'input.shape = {input.shape}')
print(f'output.shape = {output.shape}')
print(f'conv_layer.weight.shape = {conv_layer.weight.shape}')
