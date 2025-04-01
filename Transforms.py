from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 创建SummaryWriter对象，指定日志保存路径为 "logs"
writer = SummaryWriter("logs")

# 定义图像路径
img_path = "dataset/train/ants/0013035.jpg"

# 使用PIL库读取指定路径的图片
img = Image.open(img_path)

"""
实例化对象
将PIL图像/NumPy数组转为PyTorch张量，并自动归一化到[0,1]范围
"""
tensor_trans = transforms.ToTensor()

# 调用实例处理图像
tensor_img = tensor_trans(img)

# 将张量格式的图像写入TensorBoard日志（自动处理CHW格式）
writer.add_image("Tensor_img", tensor_img)

# 关闭写入器（确保数据写入磁盘）
writer.close()
