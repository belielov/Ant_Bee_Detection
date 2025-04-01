from PIL import Image # 处理图像
from torch.utils.tensorboard import SummaryWriter # 记录数据到TensorBoard
from torchvision import transforms # 图像预处理工具

# 创建SummaryWriter对象，指定日志保存路径为 "logs"
writer = SummaryWriter("logs")

# 定义图像路径
img = Image.open("dataset/train/ants/0013035.jpg")

# 打印图片信息（格式、尺寸等）
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.6, 0.6], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)          # img PIL -> resize -> img_resize PIL
img_resize = trans_totensor(img_resize) # img_resize PIL -> totensor -> img_resize tensor
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize -2
trans_resize_2 = transforms.Resize(512) # 定义Resize转换（将短边缩放到512，长边按比例调整）
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((500, 700))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

# 关闭写入器（确保数据写入磁盘）
writer.close()
