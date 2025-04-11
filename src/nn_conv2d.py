import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR10测试集
dataset = torchvision.datasets.CIFAR10(
		root="./datasets",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 创建数据加载器（批量大小64）
dataloader = DataLoader(dataset, batch_size=64)

# 定义神经网络模型（仅包含一个卷积层）
class nn_Module(nn.Module):
    def __init__(self):
        super().__init__()   # 必须调用父类构造函数，初始化基础模块
        self.conv1 = Conv2d(
            in_channels=3,   # 输入通道数，对应RGB三通道
            out_channels=6,  # 输出通道数，即卷积核数量
            kernel_size=3,   # 卷积核尺寸3x3
            stride=1,        # 滑动步长
            padding=0)       # 无填充

    def forward(self, x):
        x = self.conv1(x) # 执行卷积操作
        return x

# 实例化模型
nn_module = nn_Module()

# 创建SummaryWriter对象，日志保存在"logs"目录
writer = SummaryWriter("logs")

# 遍历数据集并记录结果
step = 0
for data in dataloader:
    imgs, targets = data
    output = nn_module(imgs)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30])
    # 强行重塑为3通道，用于可视化
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1

# 关闭写入器，确保所有数据写入磁盘
writer.close()
