import torchvision
"""
注意：如果该数据集已经下载过，需要把后面的参数修改为 download=False
"""

# 下载 MNIST 数据集
train_set_MNIST = torchvision.datasets.MNIST(root='D:\MyProject\Pytorch\Dataset\MNIST', train=True, download=False)
test_set_MNIST = torchvision.datasets.MNIST(root='D:\MyProject\Pytorch\Dataset\MNIST', train=True, download=False)

# 下载 CIFAR10 数据集
train_set_CIFAR10 = torchvision.datasets.CIFAR10(root='D:\MyProject\Pytorch\Dataset\CIFAR10', train=True, download=False)
test_set_CIFAR10 = torchvision.datasets.CIFAR10(root='D:\MyProject\Pytorch\Dataset\CIFAR10', train=True, download=False)
