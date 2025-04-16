""" 导入必要的库 """
import torch
from PIL import Image
from torchvision import transforms
from Ant_Bee_Detection_v2 import make_net

""" 配置设备 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

""" 加载模型文件 """


def load_model(model_path):
    """ 加载预训练模型权重并返回配置好的模型

    Args:
            model_path(str): 模型权重文件路径（.pth文件）

    Returns:
            model: 加载权重后的模型实例，部署在指定设备并处于评估模式
    """
    # 初始化模型结构
    model = make_net()

    # 加载保存的权重
    checkpoint = torch.load(model_path, map_location=device)

    # 将权重加载到模型实例中
    model.load_state_dict(checkpoint['state_dict'])

    # 设置为评估模式
    model.eval()

    return model


# 完成模型的完整初始化
model = load_model('best_model_v2.pth')

""" 预处理输入数据 """
# 测试集的预处理流程（必须与训练时相同）
preprocess = transforms.Compose([
    transforms.Resize(256),  # 将图像短边缩放至256像素（保持长宽比）
    transforms.CenterCrop(224),  # 从图像中心裁剪224x224区域
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(  # 使用与训练集相同的标准化参数（保持数据分布一致）
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

""" 预测函数 """


def predict_batch(image_paths, batch_size=8):
    """ 批量预测多张图像
    Args:
            image_path(list): 图像路径列表，例如 ['img1.jpg', 'img2.jpg']
            batch_size(int): 单次推理的批次大小（根据GPU内存调整）
    Returns:
            list: 预测结果列表，每个元素为（类别名称，置信度），无效图像位置返回None
    """
    batch_images = []
    valid_mask = []  # 记录哪些位置是有效的

    # 第一阶段：预处理所有图像
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')  # 确保RGB格式
            tensor = preprocess(img)
            batch_images.append(tensor)
            valid_mask.append(True)  # 标记有效位置
        except Exception as e:
            print(f'Error loading {path}: {e}')
            batch_images.append(None)  # 标记无效图像
            valid_mask.append(False)  # 标记无效位置

    results = [None] * len(image_paths)  # 初始化全为None

    # 第二阶段：分批次处理
    for i in range(0, len(batch_images), batch_size):
        # 提取当前批次，过滤无效图像
        batch = batch_images[i:i + batch_size]
        valid_indices = [j for j, img in enumerate(batch) if img is not None]
        valid_batch = [batch[j] for j in valid_indices]

        if not valid_batch:
            continue  # 跳过空批次

        # 堆叠张量并转移到设备
        input_batch = torch.stack(valid_batch).to(device)

        # 第三阶段：模型推理
        with torch.no_grad():  # 禁用梯度计算以节省内存
            outputs = model(input_batch)

            # 将logits转换为概率分布（dim=1表示按行计算）
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # 获取最大概率值及其对应索引
            _, preds = torch.max(probs, 1)

        # 第四阶段：结果映射
        class_names = ['ant', 'bee']
        for rel_idx, (abs_idx, pred) in enumerate(zip(valid_indices, preds)):
            global_idx = i + abs_idx  # 计算全局索引
            prob = probs[rel_idx][pred].item()  # 转换为Python float类型
            results[global_idx] = (class_names[pred.item()], prob)

    return results


""" 主程序 """
if __name__ == "__main__":
    image_paths = ['./dataset/test/ant_0.jpg',
                   './dataset/test/ant_1.jpg',
                   './dataset/test/ant_2.jpg',
                   './dataset/test/ant_3.jpg',
                   './dataset/test/ant_4.jpg',
                   './dataset/test/123.jpg',
                   './dataset/test/bee_0.jpg',
                   './dataset/test/bee_1.jpg',
                   './dataset/test/bee_2.jpg',
                   './dataset/test/bee_3.jpg',
                   './dataset/test/bee_4.jpg']

    results = predict_batch(image_paths, batch_size=5)

    # 打印结果
    for idx, (path, result) in enumerate(zip(image_paths, results)):
        if result is None:
            print(f'Image {idx + 1}: {path} → Failed to load')
        else:
            label, confidence = result
            print(f"Image {idx + 1}: {path} → {label} ({confidence * 100:.1f}%)")
