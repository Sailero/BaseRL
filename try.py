import torch

x = torch.tensor([0.5, -1.0, 3.0])
x_sigmoid = torch.sigmoid(x)
print(x_sigmoid)  # 输出在 [0, 1] 之间