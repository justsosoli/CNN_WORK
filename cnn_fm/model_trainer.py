"""
神经网络训练工具模块
==================

提供通用的模型训练框架，支持:
- 分类任务训练
- 回归任务训练
- 早停机制 (Early Stopping)
- 模型检查点保存 (Model Checkpoint)
- TensorBoard 可视化

典型用法:
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        task_type='classification'  # 或 'regression'
    )
    trainer.train(num_epochs=10)
    trainer.plot_curves()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# 类型定义
# =============================================================================

class TaskType(Enum):
    """任务类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class TrainerConfig:
    """
    训练器配置类
    
    用于集中管理训练相关的超参数，使代码更清晰易维护。
    """
    # 训练相关
    num_epochs: int = 10
    eval_step: int = 100  # 每隔多少 batch 执行一次验证
    
    # 早停相关
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.01
    
    # 模型保存相关
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    
    # TensorBoard 相关
    tensorboard_dir: str = "./runs"
    
    # 任务类型
    task_type: TaskType = TaskType.CLASSIFICATION


@dataclass
class TrainingHistory:
    """
    训练历史记录容器
    
    用于存储训练过程中的损失和指标，便于后续分析和可视化。
    """
    # 按 batch 记录的训练指标
    train_loss_history: list[float] = field(default_factory=list)
    train_acc_history: list[float] = field(default_factory=list)
    
    # 按 eval_step 间隔记录的验证指标
    val_loss_history: list[float] = field(default_factory=list)
    val_acc_history: list[float] = field(default_factory=list)
    
    # Epoch 级别的汇总指标
    epoch_train_loss: list[float] = field(default_factory=list)
    epoch_train_acc: list[float] = field(default_factory=list)
    epoch_val_loss: list[float] = field(default_factory=list)
    epoch_val_acc: list[float] = field(default_factory=list)


# =============================================================================
# 回调函数协议 (Protocol)
# =============================================================================

class Callback(Protocol):
    """回调函数协议基类"""
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """每个 epoch 结束时调用"""
        ...
    
    def on_batch_end(self, batch: int, logs: dict[str, Any]) -> None:
        """每个 batch 结束时调用"""
        ...


# =============================================================================
# EarlyStopping 早停机制
# =============================================================================

class EarlyStopping:
    """
    早停机制 (Early Stopping)
    
    在验证集指标在连续 patience 个 epoch 内没有改善时，提前终止训练。
    
    工作原理:
    ---------
    - mode='min': 监控指标越小越好（如 loss）
    - mode='max': 监控指标越大越好（如 accuracy）
    - 只有当指标改善量超过 min_delta 时，才认为有真正的提升
    
    示例:
        early_stopping = EarlyStopping(patience=5, min_delta=0.01, mode='min')
        
        for epoch in range(num_epochs):
            val_loss = evaluate()
            if early_stopping.step(val_loss):
                print("早停触发，停止训练")
                break
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        参数:
            patience: 容忍的 epoch 数，在此期间指标未改善则停止训练
            min_delta: 认为指标改善的最小变化量
            mode: 'min' 监控 loss（越小越好），'max' 监控 accuracy（越大越好）
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        
        # 验证 mode 参数
        if mode not in ('min', 'max'):
            raise ValueError(f"mode 必须是 'min' 或 'max'，got '{mode}'")
        
        # 根据 mode 设置比较函数
        if mode == 'min':
            self._is_improvement = lambda curr, best: curr < best - min_delta
        else:
            self._is_improvement = lambda curr, best: curr > best + min_delta
    
    def step(self, current: float) -> bool:
        """
        在每次验证后调用，检查是否应该停止训练。
        
        参数:
            current: 当前验证集的监控指标值
            
        返回:
            True 如果应该停止训练，False 否则
        """
        # 首次调用，初始化最佳分数
        if self.best_score is None:
            self.best_score = current
            self.counter = 0
            self.early_stop = False
            return False
        
        # 检查是否有改善
        if self._is_improvement(current, self.best_score):
            if self.verbose:
                improvement = self.best_score - current
                print(f"[EarlyStopping] 指标改善: {improvement:.4f}，重置计数器")
            
            self.best_score = current
            self.counter = 0
            self.early_stop = False
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] 指标未改善 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] 早停触发！连续 {self.patience} 个 epoch 无改善")
        
        return self.early_stop
    
    def reset(self) -> None:
        """重置早停状态，用于重新开始训练"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False


# =============================================================================
# ModelCheckpoint 模型检查点
# =============================================================================

class ModelCheckpoint:
    """
    模型检查点保存器
    
    功能:
    - 自动保存训练过程中的最佳模型
    - 支持按 epoch 间隔保存所有模型
    - 保存模型结构和权重
    
    示例:
        checkpoint = ModelCheckpoint(
            filepath="./checkpoints/model_{epoch:02d}_val_loss{val_loss:.4f}.pt",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        min_delta: float = 0.01,
        verbose: bool = True
    ):
        """
        参数:
            filepath: 模型保存路径，支持格式化字符串
                如 'model_{epoch}_val_loss{val_loss:.2f}.pt'
            monitor: 监控的指标名称（用于日志记录）
            save_best_only: 是否只保存最佳模型
            mode: 'min' 或 'max'，确定"更好"的含义
            min_delta: 指标改善的最小阈值
            verbose: 是否打印保存信息
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        # 初始化最佳分数
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        
        # 设置比较函数
        if mode == 'min':
            self._is_better = lambda curr, best: curr < best - min_delta
        else:
            self._is_better = lambda curr, best: curr > best + min_delta
    
    def __call__(
        self,
        current: float,
        model: nn.Module,
        epoch: Optional[int] = None
    ) -> bool:
        """
        保存模型检查点
        
        参数:
            current: 当前验证集的监控指标值
            model: 要保存的 PyTorch 模型
            epoch: 当前 epoch 编号（用于文件命名）
            
        返回:
            True 如果执行了保存操作，False 否则
        """
        # 确定保存路径
        if self.save_best_only:
            # 只保存最佳模型
            if not self._is_better(current, self.best_score):
                return False
            
            self.best_score = current
            save_path = self.filepath.format(epoch='best', **{self.monitor: current})
            
        else:
            # 每次都保存
            if epoch is not None:
                save_path = self.filepath.format(epoch=epoch, **{self.monitor: current})
            else:
                save_path = self.filepath.format(**{self.monitor: current})
        
        # 确保目录存在
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # 保存模型
        self._save_model(model, save_path)
        
        if self.verbose:
            print(f"[ModelCheckpoint] 模型已保存: {save_path}")
        
        return True
    
    def _save_model(self, model: nn.Module, filepath: str) -> None:
        """
        内部方法：保存模型到文件
        
        参数:
            model: PyTorch 模型
            filepath: 保存路径
        """
        torch.save(model.state_dict(), filepath)


# =============================================================================
# TensorBoardCallback
# =============================================================================

class TensorBoardCallback:
    """
    TensorBoard 回调类
    
    用于将训练指标实时记录到 TensorBoard，方便可视化分析。
    
    启动 TensorBoard:
        tensorboard --logdir=./runs --port=6006
    
    示例:
        tb_callback = TensorBoardCallback(log_dir="./runs/experiment_1")
        tb_callback.draw_model(model, input_shape=(1, 1, 28, 28))
        
        # 训练过程中
        tb_callback(0, loss=0.5, val_loss=0.4, acc=0.85, val_acc=0.82)
    """
    
    def __init__(
        self,
        log_dir: str,
        flush_secs: int = 30,
        verbose: bool = False
    ):
        """
        参数:
            log_dir: 日志文件保存目录
            flush_secs: 每隔多少秒将数据写入磁盘
            verbose: 是否打印详细信息
        """
        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=flush_secs
        )
        self.verbose = verbose
    
    def draw_model(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: torch.device = None
    ) -> None:
        """
        绘制模型计算图结构
        
        参数:
            model: 要可视化的模型
            input_shape: 输入张量的形状
            device: 模型所在设备
        """
        if device is None:
            device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
        
        dummy_input = torch.randn(*input_shape).to(device)
        self.writer.add_graph(model, dummy_input)
        
        if self.verbose:
            print(f"[TensorBoard] 模型图已添加到 {self.writer.log_dir}")
    
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """添加单个标量值"""
        self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag: str, tag_dict: dict[str, float], step: int) -> None:
        """添加多个标量值到同一个 main_tag"""
        self.writer.add_scalars(main_tag, tag_dict, step)
    
    def add_loss_scalars(
        self,
        step: int,
        loss: float,
        val_loss: Optional[float] = None
    ) -> None:
        """记录损失曲线"""
        tag_dict = {"train_loss": loss}
        if val_loss is not None:
            tag_dict["val_loss"] = val_loss
        self.writer.add_scalars("Loss", tag_dict, step)
    
    def add_acc_scalars(
        self,
        step: int,
        acc: float,
        val_acc: Optional[float] = None
    ) -> None:
        """记录准确率曲线"""
        tag_dict = {"train_acc": acc}
        if val_acc is not None:
            tag_dict["val_acc"] = val_acc
        self.writer.add_scalars("Accuracy", tag_dict, step)
    
    def add_lr_scalar(self, step: int, learning_rate: float) -> None:
        """记录学习率"""
        self.writer.add_scalar("Learning_Rate", learning_rate, step)
    
    def __call__(
        self,
        step: int,
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        统一回调接口，一次调用可记录多个指标
        
        参数:
            step: 当前训练步数
            loss: 当前 batch 的训练损失
            val_loss: 当前验证损失
            acc: 当前 batch 的训练准确率
            val_acc: 当前验证准确率
            lr: 当前学习率
            **kwargs: 其他要记录的标量
        """
        if loss is not None and val_loss is not None:
            self.add_loss_scalars(step, loss, val_loss)
        
        if acc is not None and val_acc is not None:
            self.add_acc_scalars(step, acc, val_acc)
        
        if lr is not None:
            self.add_lr_scalar(step, lr)
        
        # 处理其他自定义标量
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.add_scalar(f"Custom/{key}", value, step)
    
    def close(self) -> None:
        """关闭 writer，释放资源"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Trainer 训练器主类
# =============================================================================

class Trainer:
    """
    通用神经网络训练器
    
    支持分类和回归任务，提供完整的训练、验证和可视化流程。
    
    主要特性:
    - 支持分类和回归任务
    - 可配置的早停机制
    - 模型检查点自动保存
    - TensorBoard 日志记录
    - 训练曲线可视化
    
    示例 (分类任务):
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            task_type=TaskType.CLASSIFICATION
        )
        history = trainer.train(num_epochs=10)
        trainer.plot_curves()
    
    示例 (回归任务):
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            device=device,
            task_type=TaskType.REGRESSION
        )
        trainer.train(num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        task_type: TaskType | str = TaskType.CLASSIFICATION,
        eval_step: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        model_checkpoint: Optional[ModelCheckpoint] = None,
        tensorboard_callback: Optional[TensorBoardCallback] = None,
        callbacks: Optional[list[Callback]] = None,
        gradient_clip_val: Optional[float] = None,
        verbose: bool = True,
        x_axis: str = 'batch'
    ):
        """
        参数:
            model: 神经网络模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备 ('cuda' 或 'cpu')
            task_type: 任务类型，'classification' 或 'regression'
            eval_step: 每隔多少 batch 执行一次验证
            early_stopping: 早停对象，None 表示不使用早停
            model_checkpoint: 模型检查点保存器，None 表示不保存
            tensorboard_callback: TensorBoard 回调，None 表示不使用
            callbacks: 自定义回调列表
            gradient_clip_val: 梯度裁剪阈值，None 表示不裁剪
            verbose: 是否打印详细信息
            x_axis: 可视化 x 轴对齐方式。'batch' 将验证点映射到 batch 坐标；'epoch' 按 epoch 画曲线。
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.eval_step = eval_step
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.tensorboard_callback = tensorboard_callback
        self.callbacks = callbacks or []
        self.gradient_clip_val = gradient_clip_val
        self.verbose = verbose
        self.x_axis = 'epoch' if x_axis == 'epoch' else 'batch'
        
        # 解析任务类型
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        self.task_type = task_type
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 初始化训练历史
        self.history = TrainingHistory()
        
        # 早停回退：如果未传入但配置了早停参数
        if self.early_stopping is None:
            self.early_stopping = None
    
    def train(self, num_epochs: int) -> TrainingHistory:
        """
        执行完整的训练流程
        
        参数:
            num_epochs: 训练的轮数
            
        返回:
            TrainingHistory: 包含训练历史记录的对象
        """
        if self.task_type == TaskType.REGRESSION:
            return self._train_regression(num_epochs)
        else:
            return self._train_classification(num_epochs)
    
    def _train_classification(self, num_epochs: int) -> TrainingHistory:
        """分类任务训练"""
        global_step = 0
        
        for epoch in range(num_epochs):
            # 执行自定义回调
            self._trigger_callbacks('epoch_start', epoch)
            
            # 检查是否应该停止
            if self._check_early_stopping():
                break
            
            # 训练一个 epoch
            epoch_metrics = self._train_one_epoch_classification(epoch, global_step)
            global_step = epoch_metrics['global_step']
            
            # 打印 epoch 结果
            if self.verbose:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {epoch_metrics['train_loss']:.4f} "
                    f"Train Acc: {epoch_metrics['train_acc']:.4f}"
                )
            
            # 执行自定义回调
            self._trigger_callbacks('epoch_end', epoch, epoch_metrics)
        
        return self.history
    
    def _train_regression(self, num_epochs: int) -> TrainingHistory:
        """回归任务训练"""
        global_step = 0
        
        for epoch in range(num_epochs):
            if self._check_early_stopping():
                break
            
            epoch_metrics = self._train_one_epoch_regression(epoch, global_step)
            global_step = epoch_metrics['global_step']
            
            if self.verbose:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {epoch_metrics['train_loss']:.4f}"
                )
            
            self._trigger_callbacks('epoch_end', epoch, epoch_metrics)
        
        return self.history
    
    def _train_one_epoch_classification(
        self,
        epoch: int,
        start_global_step: int
    ) -> dict[str, Any]:
        """训练一个 epoch（分类任务）"""
        self.model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        global_step = start_global_step
        last_val_loss: Optional[float] = None
        last_val_acc: Optional[float] = None
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = self._move_to_device(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（如果配置了）
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
            
            self.optimizer.step()
            
            # 计算 batch 指标
            batch_loss = loss.item()
            predicted = torch.argmax(outputs, dim=1)
            batch_correct = (predicted == targets).sum().item()
            batch_total = targets.size(0)
            batch_acc = batch_correct / batch_total
            
            # 记录历史
            self.history.train_loss_history.append(batch_loss)
            self.history.train_acc_history.append(batch_acc)
            
            # 累加 epoch 统计
            train_loss += batch_loss * batch_total
            train_correct += batch_correct
            train_total += batch_total
            global_step += 1
            
            # 定期验证
            if global_step % self.eval_step == 0:
                val_loss, val_acc = self._evaluate_classification()
                
                self.history.val_loss_history.append(val_loss)
                self.history.val_acc_history.append(val_acc)
                
                last_val_loss = val_loss
                last_val_acc = val_acc
                
                if self.verbose:
                    print(
                        f"[Step {global_step}] "
                        f"Val Loss: {val_loss:.4f} "
                        f"Val Acc: {val_acc:.4f}"
                    )
                
                # 保存模型
                self._save_checkpoint(val_loss, val_acc, epoch)
                
                # TensorBoard 记录
                self._log_to_tensorboard(
                    global_step, batch_loss, val_loss, batch_acc, val_acc
                )
                
                # 早停检查
                self._check_early_stopping_step(val_loss, val_acc, global_step)
        
        # 计算 epoch 平均指标
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        
        self.history.epoch_train_loss.append(avg_train_loss)
        self.history.epoch_train_acc.append(avg_train_acc)
        
        if last_val_loss is not None:
            self.history.epoch_val_loss.append(last_val_loss)
        if last_val_acc is not None:
            self.history.epoch_val_acc.append(last_val_acc)
        
        return {
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': last_val_loss,
            'val_acc': last_val_acc,
            'global_step': global_step
        }
    
    def _train_one_epoch_regression(
        self,
        epoch: int,
        start_global_step: int
    ) -> dict[str, Any]:
        """训练一个 epoch（回归任务）"""
        self.model.train()
        
        train_loss = 0.0
        train_total = 0
        global_step = start_global_step
        last_val_loss: Optional[float] = None
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = self._move_to_device(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
            
            self.optimizer.step()
            
            # 计算 batch 指标
            batch_loss = loss.item()
            train_loss += batch_loss * inputs.size(0)
            train_total += inputs.size(0)
            
            self.history.train_loss_history.append(batch_loss)
            global_step += 1
            
            # 定期验证
            if global_step % self.eval_step == 0:
                val_loss = self._evaluate_regression()
                self.history.val_loss_history.append(val_loss)
                
                last_val_loss = val_loss
                
                if self.verbose:
                    print(f"[Step {global_step}] Val Loss: {val_loss:.4f}")
                
                # 保存模型
                self._save_checkpoint(val_loss, None, epoch)
                
                # TensorBoard
                self._log_to_tensorboard_regression(global_step, batch_loss, val_loss)
                
                # 早停检查
                self._check_early_stopping_step(val_loss, None, global_step)
        
        avg_train_loss = train_loss / train_total
        self.history.epoch_train_loss.append(avg_train_loss)
        
        if last_val_loss is not None:
            self.history.epoch_val_loss.append(last_val_loss)
        
        return {
            'train_loss': avg_train_loss,
            'val_loss': last_val_loss,
            'global_step': global_step
        }
    
    def _move_to_device(self, batch: tuple) -> tuple:
        """将 batch 数据移动到设备"""
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)
    
    def _evaluate_classification(self) -> tuple[float, float]:
        """评估分类模型"""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._move_to_device(batch)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        avg_val_loss = val_loss / total
        val_acc = correct / total
        
        return avg_val_loss, val_acc
    
    def _evaluate_regression(self) -> float:
        """评估回归模型"""
        self.model.eval()
        
        val_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._move_to_device(batch)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        return val_loss / total_samples
    
    def _save_checkpoint(
        self,
        val_loss: float,
        val_acc: Optional[float],
        epoch: int
    ) -> None:
        """保存模型检查点"""
        if self.model_checkpoint is not None:
            metric = val_acc if val_acc is not None else val_loss
            self.model_checkpoint(metric, self.model, epoch)
    
    def _log_to_tensorboard(
        self,
        step: int,
        loss: float,
        val_loss: float,
        acc: float,
        val_acc: float
    ) -> None:
        """记录到 TensorBoard"""
        if self.tensorboard_callback is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.tensorboard_callback(
                step=step,
                loss=loss,
                val_loss=val_loss,
                acc=acc,
                val_acc=val_acc,
                lr=lr
            )
    
    def _log_to_tensorboard_regression(
        self,
        step: int,
        loss: float,
        val_loss: float
    ) -> None:
        """记录回归指标到 TensorBoard"""
        if self.tensorboard_callback is not None:
            self.tensorboard_callback(
                step=step,
                loss=loss,
                val_loss=val_loss
            )
    
    def _check_early_stopping_step(
        self,
        val_loss: float,
        val_acc: Optional[float],
        global_step: int
    ) -> bool:
        """检查是否应该早停"""
        if self.early_stopping is None:
            return False
        
        metric = val_acc if val_acc is not None else val_loss
        should_stop = self.early_stopping.step(metric)
        
        if should_stop:
            if self.verbose:
                print(f"Early stopping triggered at step {global_step}")
        
        return should_stop
    
    def _check_early_stopping(self) -> bool:
        """检查是否已触发早停"""
        return self.early_stopping is not None and self.early_stopping.early_stop
    
    def _trigger_callbacks(
        self,
        event: str,
        epoch: int,
        metrics: Optional[dict] = None
    ) -> None:
        """触发自定义回调"""
        logs = {'epoch': epoch}
        if metrics:
            logs.update(metrics)
        
        for callback in self.callbacks:
            if hasattr(callback, f'on_{event}'):
                getattr(callback, f'on_{event}')(epoch, logs)
    
    # =========================================================================
    # 可视化方法
    # =========================================================================
    
    def plot_curves(
        self,
        mode: str = 'epoch',
        sample_step: int = 1,
        figsize: tuple = (14, 5),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制训练曲线
        
        参数:
            mode: 可视化模式。
                - 'epoch': 以 epoch 为 x 轴，更准确表达训练过程。
                - 'batch': 以 batch 为 x 轴，适合观察 batch 级别波动。
            sample_step: 采样步长，mode='batch' 时生效。
            figsize: 图形尺寸
            show: 是否显示图形
            save_path: 保存路径，None 则不保存
        """
        import matplotlib.pyplot as plt
        
        mode = 'epoch' if mode == 'epoch' else 'batch'
        
        if mode == 'epoch':
            self._plot_epoch_curves(figsize, show, save_path)
        else:
            self._plot_batch_curves(sample_step, figsize, show, save_path)
    
    def _plot_epoch_curves(
        self,
        figsize: tuple = (14, 5),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """按 epoch 绘制训练与验证曲线"""
        import matplotlib.pyplot as plt
        
        epochs = list(range(1, len(self.history.epoch_train_loss) + 1))
        
        has_acc = len(self.history.epoch_train_acc) > 0 and len(self.history.epoch_val_acc) > 0
        
        if has_acc:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # 损失曲线
            axes[0].plot(epochs, self.history.epoch_train_loss, 'b-o', label='Train Loss', markersize=5)
            if self.history.epoch_val_loss:
                axes[0].plot(epochs, self.history.epoch_val_loss, 'r-s', label='Val Loss', markersize=5)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 准确率曲线
            axes[1].plot(epochs, self.history.epoch_train_acc, 'b-o', label='Train Acc', markersize=5)
            if self.history.epoch_val_acc:
                axes[1].plot(epochs, self.history.epoch_val_acc, 'r-s', label='Val Acc', markersize=5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(epochs, self.history.epoch_train_loss, 'b-o', label='Train Loss', markersize=5)
            if self.history.epoch_val_loss:
                ax.plot(epochs, self.history.epoch_val_loss, 'r-s', label='Val Loss', markersize=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def _plot_batch_curves(
        self,
        sample_step: int = 1,
        figsize: tuple = (14, 5),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """按 batch 绘制训练曲线，并把验证点对齐到对应 batch 坐标"""
        import matplotlib.pyplot as plt
        
        train_loss = self.history.train_loss_history[::sample_step]
        val_loss = self.history.val_loss_history
        
        has_acc = len(self.history.train_acc_history) > 0
        
        if has_acc:
            train_acc = self.history.train_acc_history[::sample_step]
            val_acc = self.history.val_acc_history
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # 损失曲线
            axes[0].plot(train_loss, label='Train Loss', alpha=0.8)
            if val_loss:
                steps = self._map_val_indices_to_batch_steps(len(train_loss), len(val_loss))
                axes[0].plot(steps, val_loss, label='Val Loss', marker='o', markersize=3)
            axes[0].set_xlabel('Batch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 准确率曲线
            axes[1].plot(train_acc, label='Train Acc', alpha=0.8)
            if val_acc:
                steps = self._map_val_indices_to_batch_steps(len(train_acc), len(val_acc))
                axes[1].plot(steps, val_acc, label='Val Acc', marker='o', markersize=3)
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(train_loss, label='Train Loss', alpha=0.8)
            if val_loss:
                steps = self._map_val_indices_to_batch_steps(len(train_loss), len(val_loss))
                ax.plot(steps, val_loss, label='Val Loss', marker='o', markersize=3)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def _map_val_indices_to_batch_steps(self, train_len: int, val_len: int) -> list[int]:
        """
        将验证点映射到 batch 坐标。
        
        尽量按 eval_step 映射；若 history 长度不匹配，则退化为等距插值。
        """
        if val_len <= 0:
            return []
        if train_len <= 0:
            return [0] * val_len
        
        if hasattr(self, 'eval_step') and self.eval_step > 0:
            indices = [self.eval_step * (i + 1) - 1 for i in range(val_len)]
            if 0 <= indices[0] < train_len and 0 <= indices[-1] < train_len:
                return indices
        
        return [
            int(round(i * (train_len - 1) / max(val_len - 1, 1)))
            for i in range(val_len)
        ]
    
    def get_best_epoch(self) -> tuple[int, float]:
        """
        获取最佳 epoch 信息
        
        返回:
            (best_epoch, best_metric): 最佳 epoch 编号和对应的指标值
        """
        if self.history.epoch_val_loss:
            best_idx = np.argmin(self.history.epoch_val_loss)
            return best_idx + 1, self.history.epoch_val_loss[best_idx]
        elif self.history.epoch_val_acc:
            best_idx = np.argmax(self.history.epoch_val_acc)
            return best_idx + 1, self.history.epoch_val_acc[best_idx]
        else:
            return -1, float('inf')
    
    def summary(self) -> str:
        """生成训练摘要"""
        best_epoch, best_metric = self.get_best_epoch()
        
        summary = f"""
{'=' * 50}
Training Summary
{'=' * 50}
Task Type: {self.task_type.value}
Total Epochs: {len(self.history.epoch_train_loss)}
Final Train Loss: {self.history.epoch_train_loss[-1]:.4f}
Final Val Loss: {self.history.epoch_val_loss[-1] if self.history.epoch_val_loss else 'N/A'}
Best Epoch: {best_epoch} (metric: {best_metric:.4f})
{'=' * 50}
"""
        return summary


# =============================================================================
# 工厂函数
# =============================================================================

def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: TaskType | str = TaskType.CLASSIFICATION,
    config: Optional[TrainerConfig] = None,
    loss_fn: Optional[nn.Module] = None,
    **kwargs
) -> Trainer:
    """
    训练器工厂函数
    
    提供更简洁的接口来创建训练器，自动处理默认配置。
    
    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 计算设备
        task_type: 任务类型
        config: 训练配置，None 则使用默认配置
        loss_fn: 损失函数，None 则自动选择
        **kwargs: 其他参数传递给 Trainer
        
    返回:
        Trainer 实例
    """
    # 使用默认配置
    if config is None:
        config = TrainerConfig(task_type=TaskType(task_type) if isinstance(task_type, str) else task_type)
    
    # 自动选择损失函数
    if loss_fn is None:
        if config.task_type == TaskType.REGRESSION:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
    
    # 创建回调
    callbacks = []
    
    if config.checkpoint_dir:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'model_{epoch}_val_{val_loss:.4f}.pt')
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=config.save_best_only,
            mode='min'
        )
    else:
        model_checkpoint = None
    
    tensorboard_callback = None
    if config.tensorboard_dir:
        tensorboard_callback = TensorBoardCallback(
            log_dir=config.tensorboard_dir,
            verbose=False
        )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        device=device,
        task_type=config.task_type,
        eval_step=config.eval_step,
        model_checkpoint=model_checkpoint,
        tensorboard_callback=tensorboard_callback,
        **kwargs
    )
    
    return trainer


# =============================================================================
# 主程序入口（示例用法）
# =============================================================================

if __name__ == "__main__":
    # 示例：使用整理后的训练框架
    print(__doc__)
    
    # 示例配置
    print("\n示例配置用法:")
    print("-" * 40)
    
    # 基本用法
    # trainer = create_trainer(
    #     model=my_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    #     device=torch.device('cuda'),
    #     task_type='classification'
    # )
    # trainer.train(num_epochs=10)
    # trainer.plot_curves()
    
    print("请参考上述文档字符串中的示例代码")
