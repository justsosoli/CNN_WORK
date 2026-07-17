# CNN CIFAR-10 图像分类项目

基于 PyTorch 实现的卷积神经网络，用于 CIFAR-10 数据集图像分类任务。

## 项目成果

| 指标 | 数值 |
|------|------|
| **Kaggle 提交准确率** | **0.8633 (86.33%)** |
| 验证集最佳准确率 | ~86% |
| 模型参数量 | 4,756,170 |

## 项目结构

```
cnn_cifar10/
├── cifar10_work.ipynb      # 主要训练 Notebook
├── model_trainer.py        # 通用训练框架
├── data/                   # 数据目录
│   └── cifar10_data/       # CIFAR-10 数据集
│       ├── train/          # 训练图像
│       ├── test/           # 测试图像
│       └── *.csv           # 标签文件
├── competition             # 比赛
└── README.md               # 项目文档
```

## 模型架构

自定义 VGG 风格 CNN 模型 `CNN_CIFAR10`：

```
┌─────────────────────────────────────────────────────────────┐
│                    CNN_CIFAR10 架构                         │
├─────────────────────────────────────────────────────────────┤
│  Input: (batch, 3, 32, 32)                                 │
│                                                             │
│  ┌─ Conv Block 1 ─────────────────────────────────────┐    │
│  │  Conv2d(3→64, 3×3) → BatchNorm → ReLU               │    │
│  │  Conv2d(64→64, 3×3) → BatchNorm → ReLU → MaxPool    │    │
│  │  Output: 64×16×16                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─ Conv Block 2 ─────────────────────────────────────┐    │
│  │  Conv2d(64→128, 3×3) → BatchNorm → ReLU             │    │
│  │  Conv2d(128→128, 3×3) → BatchNorm → ReLU → MaxPool  │    │
│  │  Output: 128×8×8                                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─ Conv Block 3 ─────────────────────────────────────┐    │
│  │  Conv2d(128→256, 3×3) → BatchNorm → ReLU           │    │
│  │  Conv2d(256→256, 3×3) → BatchNorm → ReLU → MaxPool  │    │
│  │  Output: 256×4×4                                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─ Conv Block 4 ─────────────────────────────────────┐    │
│  │  Conv2d(256→512, 3×3) → BatchNorm → ReLU           │    │
│  │  Conv2d(512→512, 3×3) → BatchNorm → ReLU → MaxPool  │    │
│  │  Output: 512×2×2                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  AdaptiveAvgPool2d(1) → Flatten → FC(512→128) →           │
│  ReLU → Dropout(0.3) → FC(128→10)                          │
│                                                             │
│  Output: (batch, 10)                                       │
└─────────────────────────────────────────────────────────────┘
```

### 模型配置

| 配置项 | 值 |
|--------|-----|
| 卷积层数 | 8 层 (4 blocks × 2 conv) |
| 通道数 | 64 → 128 → 256 → 512 |
| 池化层 | 4 层 MaxPool + 1 层 AdaptiveAvgPool |
| Dropout | 0.3 |
| 总参数量 | 4,756,170 |

## 数据处理

### 数据集划分

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| 训练集 | 45,000 | 模型训练 |
| 验证集 | 5,000 | 超参数调优 |
| 测试集 | 300,000 | Kaggle 提交 |

### 数据增强

```python
train_transforms = Compose([
    RandomCrop(32, padding=4),    # 随机裁剪
    RandomHorizontalFlip(),        # 随机水平翻转
    ToTensor(),                   # 转换为张量
    Normalize(mean, std)          # 标准化
])

val_transforms = Compose([
    ToTensor(),
    Normalize(mean, std)
])
```

### CIFAR-10 类别

| 类别 | 索引 |
|------|------|
| airplane | 0 |
| automobile | 1 |
| bird | 2 |
| cat | 3 |
| deer | 4 |
| dog | 5 |
| frog | 6 |
| horse | 7 |
| ship | 8 |
| truck | 9 |

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| Batch Size | 64 |
| 训练设备 | CUDA GPU |
| 损失函数 | CrossEntropyLoss |

```python
from model_trainer import Trainer, EarlyStopping, ModelCheckpoint

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    device=device
)
trainer.train(num_epochs=50)
trainer.plot_curves()
```

## 环境依赖

```
matplotlib >= 3.11.0
numpy >= 2.4.6
pandas >= 3.0.3
scikit-learn >= 1.9.0
torch >= 2.12.1
torchvision >= 0.17.0
tqdm
pillow
```

## 运行方式

### 1. 训练模型

在 Jupyter Notebook 中运行 `cifar10_work.ipynb`：

```bash
jupyter notebook cifar10_work.ipynb
```

或使用 Python：

```python
from cifar10_work import *

# 训练模型
train_model()
```

### 2. 生成预测文件

```python
# 在训练完成后运行
test_ds = Cifar10Dataset("test", transforms_valid)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# 推理并保存结果
all_preds = []
CNN_CIFAR10_MODEL.eval()
with torch.no_grad():
    for data in test_loader_final:
        images = data[0]
        images = images.to(device)
        outputs = CNN_CIFAR10_MODEL(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy().tolist())


all_labels = [Cifar10Dataset.index_to_label[idx] for idx in all_preds]

import pandas as pd
# 生成带id的DataFrame
df = pd.DataFrame({
    "id": list(range(1 , len(all_labels) + 1)),
    "label": all_labels
})
df.to_csv("submission.csv", index=False)
```

## 参考资料

- [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Kaggle CIFAR-10 竞赛](https://www.kaggle.com/c/cifar-10)

## 提交记录

- **最终准确率**: 0.8633 (86.33%)
- **提交日期**: 2026-06-24
