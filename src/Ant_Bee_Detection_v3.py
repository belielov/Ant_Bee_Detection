""" 导入必要的库 """
import copy
import os
import time
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

""" 定义数据预处理步骤 """
dataset_transforms = {
    # 训练集使用随机增强提升泛化能力
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机缩放裁剪图像到224x224大小（增强模型对物体位置的鲁棒性）
        transforms.RandomHorizontalFlip(),  # 以50%概率水平翻转图像（增加数据多样性）
        transforms.RandomRotation(30),  # 随机旋转30度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),  # 将PIL图像转换为张量（数值范围自动归一化到[0,1]）
        transforms.Normalize(
            [0.485, 0.456, 0.406],  # 标准化处理（使用ImageNet数据集统计参数）
            [0.229, 0.224, 0.225]
        )
    ]),
    # 验证集使用确定性变换保证评估一致性
    'val': transforms.Compose([
        transforms.Resize(256),  # 将图像短边缩放至256像素（保持长宽比）
        transforms.CenterCrop(224),  # 从图像中心裁剪224x224区域
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(  # 使用与训练集相同的标准化参数（保持数据分布一致）
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}

""" 从本地读取数据 """

# 数据集根目录路径
data_dir = "./dataset"

# 创建数据集字典
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),  # 拼接完整路径
        dataset_transforms[x]  # 应用对应的预处理流程
    )
    for x in ['train', 'val']  # 同时处理训练集和验证集
}

# 创建数据加载器字典
dataloaders = {
    x: DataLoader(
        dataset=image_datasets[x],  # 对应的数据集对象
        batch_size=32,  # 每个batch包含4个样本
        shuffle=True if x == 'train' else False,  # 仅训练集打乱顺序
        num_workers=4  # 使用4个子进程加载数据
    )
    for x in ['train', 'val']
}

# 记录各数据集样本数量（用于后续训练日志显示）
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 获取类别名称列表（自动从train子目录结构解析）
class_names = image_datasets['train'].classes

# 检测并选择计算设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" 构建卷积神经网络 """


def make_net():
    """
    构建一个经典的卷积神经网络（CNN）结构，适用于图像分类任务（如二分类）
    网络结构参考LeNet的改进版本，包含3个卷积层和2个全连接层
    返回已部署到指定设备（GPU/CPU）的模型

    结构说明：
    [Conv2d → MaxPool → ReLU] × 2 → Conv2d → AdaptiveAvgPool → Flatten → Linear × 2
    输入尺寸要求：至少 32x32 像素的RGB图像（通道数3）
    输出维度：2（对应二分类任务的logits）
    """
    model = nn.Sequential(
        # 第一卷积块（提取底层特征）
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 输入3通道，输出6通道，3x3卷积核
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),  # 激活函数（原地计算节省内存）
        nn.MaxPool2d(kernel_size=2),  # 2x2最大池化，步长2（特征图尺寸减半）

        # 第二卷积块（提取中层特征）
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(2),  # 再次池化降维

        # 第三卷积块（捕获高层语义特征）
        nn.Conv2d(64, 128, 3, padding=1),  # 最终卷积层输出120通道
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.MaxPool2d(2),

        # 过渡到全连接层
        nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化到1x1（替代全局平均池化）
        nn.Flatten(),  # 展平维度（批大小×120）

        # 分类头
        nn.Linear(in_features=128, out_features=64),  # 全连接层（120 → 84维特征）
        nn.Dropout(0.5),  # 防止过拟合
        nn.ReLU(True),
        nn.Linear(in_features=64, out_features=2)  # 最终输出层（84 → 2类logits）
    )
    return model.to(device)  # 将模型部署到指定设备（GPU/CPU）


""" 设置训练参数 """


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """ 模型训练函数
    Args:
        model: 要训练的模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮次（默认25）
    Returns:
        训练完成的最佳模型
    """
    since = time.time()  # 记录训练开始时间

    # 初始化最佳模型权重和准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    lrs = []  # 存储每轮的学习率（用于后续可视化）

    # 初始化存储列表
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 开始epoch循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 新增变量保存验证指标
        val_acc = None

        # 每个epoch有两个阶段：训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            # 初始化统计指标
            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            # 使用tqdm包装数据加载器（显示进度条）
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} epoch {epoch}'):
                # 将数据移动到指定设备（如GPU）
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度（避免梯度累积）
                optimizer.zero_grad()

                # 前向传播
                # 训练时跟踪梯度，验证时不跟踪
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 获取预测类别
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才反向传播 + 优化
                    if phase == 'train':
                        loss.backward()  # 反向计算梯度
                        optimizer.step()  # 更新权重

                # 统计累计损失和正确数
                running_loss += loss.item() * inputs.size(0)  # loss.item()返回标量值
                running_corrects += torch.sum(preds == labels.data).to(device)  # 确保张量位于正确的设备上


            # 计算epoch平均指标
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 保存验证准确率
            if phase == 'val':
                val_acc = epoch_acc.item()

            # 将当前phase的指标存入列表
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())  # 转换为Python数值
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝防止引用

        # 每个epoch结束后更新学习率
        scheduler.step(val_acc)  # 关键修改
        lrs.append(optimizer.param_groups[0]['lr'])  # 关键修改

        print()  # 每个epoch后换行

    # 记录训练耗时并输出最佳验证精度
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 绘制学习率曲线
    plot_learning_rate(lrs)

    # 绘制损失和准确率曲线
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs)

    return model


""" 可视化学习率曲线 """


def plot_learning_rate(lrs):
    """ 绘制学习率变化曲线图

    Args:
        lrs (list): 包含每个epoch学习率的列表，形如 [0.1, 0.01, 0.001...]

    Returns:
        无返回值，直接保存图片到当前目录
    """
    # 创建新的画布（避免与之前图表重叠）
    plt.figure()

    # 绘制学习率曲线
    plt.plot(lrs, label='Learning Rate')

    # 设置图表元信息
    plt.title('Learning Rate Over Time')  # 标题
    plt.xlabel('Epoch')                   # x轴标签（训练轮次）
    plt.ylabel('Learning Rate')           # y轴标签（学习率数值）
    plt.legend()                          # 显示图例

    # 保存图表到本地文件
    plt.savefig('learning_rate_curve_r3.png', dpi=300, bbox_inches='tight')

    # 关闭图表，释放内存
    plt.close()


""" 可视化损失及准确率变化 """


def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs):
    """ 可视化训练过程中的损失和准确率变化曲线

    Args:
        train_losses (list): 训练集损失值列表，每个元素代表一个epoch的损失
        val_losses (list): 验证集损失值列表，需与train_losses长度一致
        train_accs (list): 训练集准确率列表，取值范围建议[0.0, 1.0]或[0, 100]
        val_accs (list): 验证集准确率列表，需与train_accs长度一致

    Returns:
        无返回值，直接在当前目录生成 loss_accuracy_curve.png 图片文件
    """
    # 生成epoch序号
    epochs = range(len(train_losses))

    # 创建画布
    plt.figure()

    # ------------------ 损失子图 ------------------
    plt.subplot(2, 1, 1)  # 2行1列的第1个子图
    # 绘制训练/验证损失曲线
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')  # 蓝色实线
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')  # 红色实线
    # 设置坐标轴信息
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()

    # ------------------ 准确率子图 ------------------
    plt.subplot(2, 1, 2)  # 2行1列的第2个子图
    # 绘制训练/验证准确率曲线
    plt.plot(epochs, train_accs, 'b--', label='Training Accuracy')  # 蓝色虚线
    plt.plot(epochs, val_accs, 'r--', label='Validation Accuracy')  # 红色虚线
    # 设置坐标轴信息
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 调整子图间距（防止标签重叠）
    plt.tight_layout()

    # 保存图片
    plt.savefig('loss_accuracy_curve_r3.png', dpi=300, bbox_inches='tight')

    # 关闭图表，释放内存
    plt.close()

    # 注：在Jupyter等交互环境直接显示，可添加 plt.show() 代替保存


""" 主程序 """
if __name__ == "__main__":
    # ------------------ 初始化模型 ------------------
    model = make_net()

    # ------------------ 定义损失函数 ------------------
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # ------------------ 定义优化器 ------------------
    optimizer = optim.Adam(
        model.parameters(),  # 需要优化的模型参数
        lr=3e-4,            # 初始学习率
        weight_decay=1e-4
    )

    # ------------------ 定义学习率调度器 ------------------
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 根据验证准确率调整
        patience=3,  # 3个epoch无提升则降低LR
        factor=0.5
    )

    # ------------------ 执行模型训练 ------------------
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=50)

    # ------------------ 保存最佳模型 ------------------
    torch.save({'model': model, 'state_dict': model.state_dict()}, 'best_model_v3.pth')
