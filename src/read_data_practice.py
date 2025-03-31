from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,image_dir,label_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.Imagepath = os.path.join(self.root_dir, self.image_dir)
        self.Labelpath = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.Imagepath)
        self.label_path = os.listdir(self.Labelpath)



    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img = Image.open(img_item_path)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        with open(label_item_path, 'r') as f:
            label = f.read().strip()  # 以只读模式'r-read'打开文本文档，并返回文件内容（去除首尾空格）
        return img, label

    def __len__(self):
        return len(self.img_path)

    def __add__(self,other):
        """ 实现数据集加法操作 """
        return ConcatDataset([self, other])

root_dir = "dataset_practice/train"
ants_image_dir = "ants_image"
ants_label_dir = "ants_label"
bees_image_dir = "bees_image"
bees_label_dir = "bees_label"
ants_dataset = MyData(root_dir, ants_image_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_image_dir, bees_label_dir)

train_dataset = ConcatDataset([ants_dataset, bees_dataset])
