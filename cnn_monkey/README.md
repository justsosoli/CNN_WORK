# CNN Monkey Classification

使用 PyTorch 实现的猴子图像10分类项目，支持自定义CNN模型和预训练ResNet模型迁移学习。

## 项目概述

本项目实现了两个图像分类模型用于识别10种不同的猴子：

- **CNN_monkey**：自定义卷积神经网络
- **ResNet**：基于预训练ResNet50的迁移学习模型

## 数据集

使用的数据集位于 `data/monkey_10` 目录下：

```
data/monkey_10/
├── training/       # 训练集 (1097 样本)
└── validation/    # 验证集 (272 样本)
```

每种类别存放在独立子文件夹中。

## 项目结构

```
cnn_monkey/
├── monkey_10.ipynb    # 主要训练
├── model_trainer.py    # 通用训练框架
└── data/              # 数据集目录
```

## 模型架构

### CNN_monkey

自定义CNN模型，包含4个卷积块：

| 层 | 输入通道 | 输出通道 | 卷积核 | 池化 |
|---|---------|---------|--------|-----|
| conv1_block | 3 | 32 | 3x3 | MaxPool 2x2 |
| conv2_block | 32 | 64 | 3x3 | MaxPool 2x2 |
| conv3_block | 64 | 128 | 3x3 | MaxPool 2x2 |
| conv4_block | 128 | 256 | 3x3 | MaxPool 2x2 |

分类头：`AdaptiveAvgPool2d → Flatten → Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→10)`

### ResNet50 (迁移学习)

使用 torchvision 预训练的 ResNet50，去除原始全连接层后接新的分类头。

## 训练配置

### 数据增强

**训练集**：
- RandomResizedCrop(224)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(10°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
- Normalize (基于数据集均值和标准差)

**验证集**：
- Resize(256) → CenterCrop(224)
- Normalize

### 训练参数

- 批量大小：64
- 优化器：Adam (学习率 0.001)
- 损失函数：CrossEntropyLoss
- 设备：CUDA (GPU)

## 实验结果

| 模型 | 训练准确率 | 验证准确率 |
|------|-----------|-----------|
| CNN_monkey | 44.44% | 56.25% |
| ResNet50 (迁移学习) | 77.78% | **97.79%** |

ResNet通过迁移学习利用ImageNet预训练权重，显著提升了分类性能。

## 环境依赖

```
matplotlib >= 3.11.0
numpy >= 2.4.6
pandas >= 3.0.3
scikit-learn >= 1.9.0
torch >= 2.12.1
torchvision
tqdm
```

## 使用方法

### 在 Jupyter Notebook 中运行

打开 `monkey_10.ipynb` 按顺序执行所有单元格即可完成数据加载、模型训练和评估。

### Python 脚本训练

```python
from model_trainer import Trainer, TaskType
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = CNN_monkey(num_classes=10)
model = model.to(device)

# 配置训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

history = trainer.train(num_epochs=20)
trainer.plot_curves()
```

## model_trainer.py 模块

通用训练框架，支持：

- 分类/回归任务
- 早停机制 (Early Stopping)
- 模型检查点保存
- TensorBoard 可视化
- 训练曲线绘制
