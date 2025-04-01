import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

""" 将图片数据写入TensorBoard日志文件 """

writer = SummaryWriter("logs")
img_path = "dataset/train/bees/16838648_415acd9e3f.jpg"

# 使用PIL库读取指定路径的图片
img_PIL = Image.open(img_path)

img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')
# # y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)

writer.close()