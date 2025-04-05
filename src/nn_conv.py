# 导入 PyTorch 库及其神经网络函数模块
import torch
import torch.nn.functional as F

# 定义输入张量（5x5矩阵）
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 定义卷积核（3x3矩阵）
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

"""
将输入和卷积核调整为 PyTorch 卷积要求的四维格式：
(batch_size, input_channels, height, width)
"""
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 第一次卷积操作：stride=1(默认)，无padding
output = F.conv2d(input, kernel, stride=1)
print("stride=1, no padding:\n", output)

# 第二次卷积操作：stride=2
output2 = F.conv2d(input, kernel, stride=2)
print("stride=2, no padding:\n", output2)

# 第三次卷积操作：stride=1，padding=1（在输入周围填充一圈0）
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print("stride=1, padding=1:\n", output3)
