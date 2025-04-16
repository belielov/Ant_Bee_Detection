import torch
from PIL import Image
from torchvision import transforms
from Ant_Bee_Detection import make_net

""" 配置设备 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

""" 加载模型文件 """


def load_model(model_path):
    """加载预训练模型权重并返回配置好的模型

    Args:
        model_path (str): 模型权重文件路径（.pth文件）

    Returns:
        module: 加载权重后的模型实例，部署在指定设备（GPU/CPU）并处于评估模式

    Note:
        - 不需要外部实例化模型，该函数内部通过 make_net() 自动构建模型结构
        - 必须保证当前代码中的 make_net() 与训练时的模型结构完全一致
    """
    # 初始化模型结构
    model = make_net()

    # 加载保存的权重（自动处理设备映射）
    # map_location=device 确保权重加载到当前可用设备（GPU/CPU
    checkpoint = torch.load(model_path, map_location=device)

    # 将权重加载到模型实例中（严格匹配层结构）
    # load_state_dict 会将权重字典注入到模型参数中
    model.load_state_dict(checkpoint['state_dict'])

    # 设置为评估模式
    model.eval()

    return model


# 完成模型的完整初始化
model = load_model('best_model.pth')

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
    """批量预测多张图像
    Args:
        image_paths (list): 图像路径列表，例如 ['img1.jpg', 'img2.jpg']
        batch_size (int): 单次推理的批次大小（根据GPU内存调整）
    Returns:
        list: 预测结果列表，每个元素为 (类别名称, 置信度)，无效图像位置返回None
    """
    batch_images = []

    # 第一阶段：预处理所有图像
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')  # 确保RGB格式
            tensor = preprocess(img)
            batch_images.append(tensor)
        except Exception as e:
            print(f'Error loading {path}: {e}')
            batch_images.append(None)  # 标记无效图像

    results = []

    # 第二阶段：分批次推理
    for i in range(0, len(batch_images), batch_size):
        # 提取当前批次，过滤无效图像
        batch = batch_images[i:i+batch_size]
        valid_indices = [j for j, img in enumerate(batch) if img is not None]
        valid_tensors = [batch[j] for j in valid_indices]

        if not valid_tensors:
            continue  # 跳过空批次

        # 堆叠张量并转移到设备
        input_batch = torch.stack(valid_tensors).to(device)

        # 第三阶段：模型推理
        with torch.no_grad():  # 禁用梯度计算以节省内存
            outputs = model(input_batch)

            # 将logits转换为概率分布（dim=1表示按行计算）
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # 获取最大概率值及其对应索引
            _, preds = torch.max(probs, 1)

        # 第四阶段：结果映射
        class_names = ['ant', 'bee']
        for idx, pred in enumerate(preds):
            original_idx = i + valid_indices[idx]  # 计算在原始列表中的位置
            prob = probs[idx][pred].item()  # 转换为Python float类型
            results.append((class_names[pred.item()], prob))

    return results


def predict_single(image_path):
    """对单张图像进行预测，返回类别名称和置信度

    Args:
        image_path (str): 待预测图像的路径

    Returns:
        tuple: (类别名称, 置信度)，例如 ('ant', 0.987)

    Raises:
        FileNotFoundError: 如果图像路径不存在
        PIL.UnidentifiedImageError: 如果图像无法被解析
    """
    # 打开图像并强制转换为RGB格式（兼容PNG/JPG等格式，避免Alpha通道干扰）
    img = Image.open(image_path).convert('RGB')

    # 应用预处理流程（必须与训练时的val_transform完全一致）
    # unsqueeze(0) 增加批次维度：从 [C, H, W] -> [1, C, H, W]
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 禁用梯度计算（提升推理速度，减少显存占用）
    with torch.no_grad():
        # 模型前向传播（直接输出logits）
        output = model(input_tensor)

        # 将logits转换为概率分布（dim=0表示对第一个维度进行softmax）
        # output[0] 是因为输入是单样本，output形状为 [1, num_classes]
        prob = torch.nn.functional.softmax(output[0], dim=0)

    # 获取最大概率对应的类别索引（item()将Tensor转为Python标量）
    pred_class = torch.argmax(prob).item()

    # 映射到类别名称（注意：class_names必须与训练时的顺序严格一致）
    # 例如训练时第一个类别是ant，此处应为 ['ant', 'bee']
    return ['ant', 'bee'][pred_class], prob[pred_class].item()


""" 单张图片使用示例
try:
    label, confidence = predict_single("test_image.jpg")
    print(f"预测结果: {label} (置信度: {confidence * 100:.1f}%)")
except FileNotFoundError:
    print("错误：图像文件不存在")
except PIL.UnidentifiedImageError:
    print("错误：无法解析图像文件")
"""

if __name__ == "__main__":
    image_paths = ['./dataset/test/ant_0.jpg',
                   './dataset/test/ant_1.jpg',
                   './dataset/test/ant_2.jpg',
                   './dataset/test/ant_3.jpg',
                   './dataset/test/ant_4.jpg',
                   './dataset/test/bee_0.jpg',
                   './dataset/test/bee_1.jpg',
                   './dataset/test/bee_2.jpg',
                   './dataset/test/bee_3.jpg',
                   './dataset/test/bee_4.jpg']

    results = predict_batch(image_paths, batch_size=5)

    # 打印结果
    for idx, (path, result) in enumerate(zip(image_paths, results)):
        if result is None:
            print(f"Image {idx + 1}: {path} → Failed to load")
        else:
            label, confidence = result
            print(f"Image {idx + 1}: {path} → {label} ({confidence * 100:.1f}%)")
