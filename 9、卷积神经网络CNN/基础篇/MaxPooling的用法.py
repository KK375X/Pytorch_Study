import torch

input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6,
         ]

input = torch.Tensor(input).view(1, 1, 4, 4)

# kernel_size=2 这样设置之后，默认 stride 也会等于 2
MaxPooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = MaxPooling_layer(input)
print(output)

"""
MaxPool2d：求出分块后每个小块中的最大值，然后拼在一起
"""
