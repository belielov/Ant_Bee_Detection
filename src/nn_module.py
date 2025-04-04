import torch         # 导入 PyTorch 库
from torch import nn # 从 PyTorch 中导入神经网络模块

# 定义一个自定义神经网络模块，继承自 nn.Module
class nn_Module(nn.Module):
    def __init__(self):
        super().__init__() # 必须调用父类构造函数，初始化基础模块
    # 定义前向传播逻辑
    def forward(self, input):
        output = input + 1 # 核心操作：输入值加 1
        return output
# 实例化自定义模块
nn_module = nn_Module()

# 创建一个值为 1.0 的标量张量
x = torch.tensor(1.0)

# 使用模块处理输入数据（自动调用 forward 方法）
output = nn_module(x)

print(output)