from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1. transforms 该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()

print(tensor_img)