import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

""" 将图片数据写入TensorBoard日志文件 """

# 创建SummaryWriter对象，指定日志保存路径为 "logs"
writer = SummaryWriter("logs")

# 定义图像路径
img_path = "dataset/train/bees/16838648_415acd9e3f.jpg"

# 使用PIL库读取指定路径的图片
img_PIL = Image.open(img_path)

# 将图片转化为numpy数组格式
img_array = np.array(img_PIL)

# 打印数组类型和形状（供调试）
print(type(img_array))
print(img_array.shape)

# 通过TensorBoard的SummaryWriter将图像数据写入日志
# 通过 tensorboard --logdir=logs 命令可视化图像
writer.add_image("test", img_array, 2, dataformats='HWC')

# 关闭写入器（确保数据写入磁盘）
writer.close()