# CNN-FM

FashionMNIST 图像分类实验项目，使用 PyTorch 实现标准卷积神经网络与深度可分离卷积网络的训练、验证和对比分析。

## 项目结构

```
cnn_fm/
├── FashionMNIST_work.ipynb    # 主实验 notebook（模型、训练、可视化）
├── model_trainer.py           # 通用训练工具（早停、检查点、TensorBoard）
└── README.md                  # 项目说明
```

## 环境依赖

- Python 3.8+
- PyTorch 2.12+ / torchvision 0.17+
- matplotlib、numpy、pandas、scikit-learn、tqdm

安装：

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn tqdm tensorboard
```

## 快速开始

在 notebook 中依次运行数据加载、模型定义、训练与对比实验：

```bash
jupyter notebook FashionMNIST_work.ipynb
```

或使用训练器快速搭建训练流程：

```python
import model_trainer as mt
import torch.nn as nn
import torch.optim as optim

trainer = mt.Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=1e-3),
    device=device,
    eval_step=100,
)

history = trainer.train(num_epochs=10)
trainer.plot_curves(mode='epoch')
```

## 实验设置

- 数据集：FashionMNIST（训练 60000 / 测试 10000，28×28 灰度）
- 批次大小：128
- 训练轮数：10
- 优化器：Adam，学习率 0.001
- 验证策略：每 100 个训练 step 在测试集上评估一次
- 设备：CUDA

## 模型与结果

### 标准卷积网络

```
Input (1×28×28)
├── Conv Block 1: Conv2d(1→32) + BN + ReLU + Conv2d(32→32) + BN + MaxPool
├── Conv Block 2: Conv2d(32→64) + BN + ReLU + Conv2d(64→64) + BN + MaxPool
└── Classifier: Flatten → Linear(64×7×7→128) → ReLU → Dropout → Linear(128→10)
```

- 参数量：468,202
- 最终验证准确率：92.66%
- 最终验证损失：0.2919
- 最终训练准确率：98.12%

### 深度可分离卷积网络

```
Input (1×28×28)
├── DSConv Block 1: DW→BN→ReLU→PW→BN→MaxPool  (1→32)
├── DSConv Block 2: DW→BN→ReLU→PW→BN→MaxPool  (32→64)
├── DSConv Block 3: DW→BN→ReLU→PW→BN→MaxPool  (64→128)
└── Classifier: Flatten → Linear(128×3×3→128) → ReLU → Dropout → Linear(128→10)
```

- 参数量：160,661
- 最终验证准确率：92.74%
- 最终验证损失：0.3982
- 最终训练准确率：99.25%

### 对比结论

| 指标 | Standard Conv | Depthwise Separable |
|------|---------------|---------------------|
| 参数量 | 468,202 | 160,661 |
| 参数量减少 | — | 65.7% |
| 最终验证 Acc | 92.66% | 92.74% |
| 最终验证 Loss | 0.2919 | 0.3982 |
| 最终训练 Acc | 98.12% | 99.25% |

在相同训练配置下，深度可分离卷积仅用约 34.3% 的参数量就达到了接近的验证准确率；标准卷积在最终验证损失上更低，说明其泛化边界略优。

## 可复用工具

`model_trainer.py` 提供通用训练框架：

- 分类 / 回归任务训练
- EarlyStopping 早停
- ModelCheckpoint 最优模型保存
- TensorBoardCallback 日志记录
- 训练历史与曲线可视化
