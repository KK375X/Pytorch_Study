import torch
import visdom

# 创建一个 Visdom 实例
vis = visdom.Visdom()

# 示例数据
x = torch.arange(0, 10, 0.1)
y = torch.sin(x)

# 绘制折线图
vis.line(X=x, Y=y, opts=dict(title='Sin Function', xlabel='x', ylabel='sin(x)'))

# 绘制散点图
vis.scatter(X=torch.randn(20, 2))

# 绘制柱状图
vis.bar(X=torch.tensor([3, 5, 2, 7, 4]))

# 关闭 Visdom 客户端连接
vis.close()
