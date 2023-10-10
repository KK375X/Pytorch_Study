import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]

# 构造输入数据，view(1, 1, 5, 5) 表示输出通道数为 1，输入通道数为 1，大小为 5*5
input = torch.Tensor(input).view(1, 1, 5, 5)
print(f'input = {input}')

# padding=1 表示输出层的宽和高增加 1，bias=False 表示本模型不需要增加偏执量
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

# 构造卷积核，view(1, 1, 3, 3) 表示输出通道数为 1，输入通道数为 1，大小为 3*3
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
print(f'kernel = {kernel}')

# 把自定义的卷积层的参数给卷积核，初始化
conv_layer.weight.data = kernel.data

# 喂入输入数据，并输出
output = conv_layer(input)
print(f'output = {output}')

"""
由于卷积层是 3*3，输入层是 5*5，并且在 padding=1 之后，输入层外围增加了一圈（上下各增加一行，左右各增加一列），导致输入层变为 7*7
因此输出层的大小为 5*5
"""
