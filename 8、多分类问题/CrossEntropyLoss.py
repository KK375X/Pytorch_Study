import torch

# 创建 交叉熵激活函数
criterion = torch.nn.CrossEntropyLoss()

# 结果值
Y = torch.LongTensor([2, 0, 1])

# 预测值
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])
Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])
# 由上面的结果值和预测值可以观察得到，Y_pred1 经过 Softmax 之后与结果值是符合的

loss1 = criterion(Y_pred1, Y)
loss2 = criterion(Y_pred2, Y)

# 由输出结果可知，Y_pred1 样本的损失更小
print("Batch Loss1 = ", loss1.data)
print("Batch Loss2 = ", loss2.data)
