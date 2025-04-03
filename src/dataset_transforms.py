# 导入torchvision库和TensorBoard的SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 定义数据预处理转换流程，将PIL图像转换为Tensor
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()  # 将图像数据转换为PyTorch Tensor格式
])

# 下载并加载CIFAR10训练集和测试集
# 参数说明：
# root: 数据集存储路径
# train: True表示训练集，False表示测试集
# transform: 应用于数据的转换操作
# download: 如果本地不存在则从网络下载
train_set = torchvision.datasets.CIFAR10(root="./datasets", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=dataset_transform, download=True)

# 显示测试集第一个样本的信息
# print(test_set[0])                 # 输出(图像Tensor, 类别索引)
# print(test_set.classes)            # 输出类别名称列表
# img, target = test_set[0]          # 获取第一个样本的图像和类别
# print(img)                         # 输出图像Tensor
# print(target)                      # 输出类别索引(0-9)
# print(test_set.classes[target])    # 输出对应的类别名称
# img.show()                         # 显示图像（需要GUI环境）

# 创建TensorBoard的SummaryWriter对象，日志保存在"p10"目录
writer = SummaryWriter("p10")

# 将测试集的前10张图片写入TensorBoard
for i in range(10):
    img, target = test_set[i]  # 获取第i个样本
    writer.add_image("test_set", img, i)  # 参数说明：标签名，图像Tensor，全局步数

# 关闭SummaryWriter，确保所有数据写入磁盘
writer.close()