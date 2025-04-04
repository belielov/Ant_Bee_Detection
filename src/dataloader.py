import torchvision # 提供流行的数据集、模型架构和图像转换
from torch.utils.data import DataLoader # 数据加载器，用于批量加载数据
from torch.utils.tensorboard import SummaryWriter # TensorBoard 可视化工具

# 准备的测试数据集
# transform将PIL图像或numpy数组转换为Tensor，并归一化到[0,1]
test_data = torchvision.datasets.CIFAR10("./datasets", train=False, transform=torchvision.transforms.ToTensor())

""" 
创建测试集的数据加载器
参数：
    dataset：使用的数据集
    batch_size：每个批次加载的图片数量
    shuffle：每个epoch打乱数据顺序（通常测试集shuffle=False，本代码仅为演示）
    num_workers：加载数据的子进程数，0表示在主进程中加载
    drop_last：如果数据总数不能被batch_size整除，丢弃最后一个不完整的batch
"""
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 获取测试数据集中第一个样本的图片和标签
img, target = test_data[0]

# 输出图片形状，应为torch.Size([3, 32, 32])(通道、高度、宽度)
print(img.shape)

# 输出标签，对应CIFAR10中的类别索引（0-9）
print(target)

# 创建SummaryWriter对象，日志保存在"dataloader"目录
writer = SummaryWriter("dataloader")

# 进行2个epoch的循环（通常测试只需1个epoch，此处为演示shuffle=True写入TensorBoard）
for epoch in range(2):
    step = 0 # 步骤计数器，用于记录每个batch在TensorBoard中的位置
    # 遍历测试数据加载器中的每个batch
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1
# 关闭写入器，确保所有数据写入磁盘
writer.close()