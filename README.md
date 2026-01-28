# CausalAir: Causal Spatio-Temporal Air Quality Prediction Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 简介

**CausalAir (Causal Air Quality Prediction Network，因果空气质量预测网络)** 是一个用于城市空气质量预测的深度学习模型，通过学习离散因果矩阵实现可解释的时空序列预测。



## 🏗️ 模型架构

### 整体架构

```
Input: 
  - 历史空气质量数据 [B, N, L, 7]
  - 历史气象数据 [B, N, L, 56]
  - 未来气象预报 [B, N, P, 56]

        ↓
    
[嵌入层] → 历史表示 + 未来表示
        
        ↓
        
[因果矩阵生成]
  ├─ 站点级因果矩阵 [N, N] (Gumbel-Softmax)
  └─ 变量级因果矩阵 [N, 7, 56] (Gumbel-Softmax)
        
        ↓
        
[CausalAir编码器] (L 层)
  ├─ 站点维度自注意力 (带因果掩码)
  ├─ 变量维度自注意力
  ├─ 气象特征更新 (带变量级因果掩码)
  ├─ 历史-未来交叉注意力
  └─ 前馈网络
        
        ↓
        
[预测头]
  ├─ 初始预测 (从历史表示)
  ├─ 最终预测 (从未来表示)
  └─ 重构损失 (历史重构)

Output: 未来48小时空气质量预测 [B, N, 48, 7]
```

### 关键组件

#### 1. 因果矩阵生成

```python
# 站点级因果矩阵
theta: [N, N, 2] → Gumbel-Softmax → causal_matrix: [N, N]

# 变量级因果矩阵  
station_var_theta: [N, 7, 56, 2] → Gumbel-Softmax → var_causal_matrix: [N, 7, 56]
```

#### 2. 后门替换机制

对于带因果矩阵的注意力计算：

```
O = (A ⊙ C) @ V + (A ⊙ (1-C)) @ Z
```

其中：
- `A`: 注意力权重矩阵
- `C`: 因果矩阵 (0或1)
- `V`: 原始值矩阵
- `Z`: 噪声矩阵

#### 3. 温度退火

```python
# 指数衰减
τ(t) = τ_final + (τ_initial - τ_final) * exp(-5 * progress)

# 线性衰减
τ(t) = τ_initial - (τ_initial - τ_final) * progress
```

---

## 📂 项目结构

```
CausalAir/
├── base/                      # 基础类定义
│   ├── base_data_loader.py   # 数据加载器基类
│   ├── base_model.py          # 模型基类
│   └── base_trainer.py        # 训练器基类
├── config/                    # 配置文件
│   └── cstpnet.json          # 模型配置
├── data_loader/              # 数据加载模块
│   ├── sts_loader.py         # 时空序列数据加载器
│   └── dataset.py            # 数据集定义
├── layers/                   # 网络层定义
│   ├── CausalAir_EncDec.py   # CausalAir编码器层
│   ├── Embed.py              # 嵌入层
│   ├── SelfAttention_Family.py  # 注意力机制家族
│   └── ...
├── model/                    # 模型定义
│   └── CausalAir.py         # CausalAir主模型
├── trainer/                  # 训练逻辑
│   └── trainer.py           # 训练器实现
├── evaluation/              # 评估指标
│   └── metric.py           # 评价指标
├── logger/                  # 日志模块
├── utils/                   # 工具函数
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── requirements.txt         # 依赖包
└── README.md               # 本文件
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12.1
- CUDA >= 10.2 (推荐使用GPU训练)

### 安装步骤

1. **克隆项目**
```bash
cd /mnt/hyin/workspace/
# 项目已存在于 CSTPNet 目录
```

2. **安装依赖**
```bash
cd CausalAir
pip install -r requirements.txt
```

3. **准备数据**

数据应该放置在以下路径：
```
/mnt/hyin/Datasets/china_stations_data/aq_level_mete/
├── train_data.pkl   # 训练数据
├── val_data.pkl     # 验证数据
└── test_data.pkl    # 测试数据
```

数据格式：
- `aq_train_data`: [B, T, N, 7] - 空气质量数据（7个特征）
- `mete_train_data`: [B, T, N, 56] - 气象数据（56个特征）
- `AQStation_coordinate`: [N, 2] - 站点坐标

### 训练模型

```bash
# 使用默认配置训练
python train.py --config config/causalair.json

# 使用多GPU训练
python train.py --config config/causalair.json --device 0,1

# 从检查点恢复训练
python train.py --config config/causalair.json --resume checkpoints/path/to/checkpoint.pth
```

### 测试模型

```bash
# 测试模型
python test.py --resume checkpoints/causalair_causal_air_quality_prediction/train/XXXXXX/model_best.pth

# 测试并保存预测结果
python test.py --resume checkpoints/path/to/model_best.pth --save_output
```

### 监控训练

使用 TensorBoard 监控训练过程：

```bash
tensorboard --logdir=checkpoints/causalair_causal_air_quality_prediction/train/
```

---

## ⚙️ 配置说明

主要配置参数（`config/causalair.json`）：

### 模型参数

```json
{
  "seq_len": 48,           // 历史序列长度（小时）
  "pred_len": 48,          // 预测序列长度（小时）
  "d_model": 256,          // 模型维度
  "n_heads": 8,            // 注意力头数
  "e_layers": 3,           // 编码器层数
  "d_ff": 1024,            // 前馈网络维度
  "dropout": 0.1,          // Dropout率
  "n_station": 1628,       // 站点数量
  "gat_node_features": 7,  // 空气质量特征数
  "mete_features": 56      // 气象特征数
}
```

### 因果学习参数

```json
{
  "sparsity_weight": 5,    // 稀疏性正则化权重
  "backdoor_attention": {
    "enabled": true,       // 启用后门注意力
    "noise_type": "uniform",  // 噪声类型
    "noise_std": 0.1,      // 噪声标准差
    "temperature": 1.0     // 初始温度
  }
}
```

### 温度调度器

```json
{
  "temperature_scheduler": {
    "type": "ExponentialDecay",  // 衰减类型：ExponentialDecay/LinearDecay
    "args": {
      "initial_temp": 1.0,       // 初始温度
      "final_temp": 0.1,         // 最终温度
      "decay_epochs": 50,        // 衰减周期
      "decay_type": "exponential"  // 衰减方式
    }
  }
}
```

### 训练参数

```json
{
  "optimizer": {
    "type": "Adam",
    "args": {
      "lr": 0.0001,        // 学习率
      "weight_decay": 0,   // 权重衰减
      "amsgrad": true
    }
  },
  "trainer": {
    "epochs": 100,         // 训练轮数
    "save_period": 10,     // 保存周期
    "early_stop": 50,      // 早停轮数
    "monitor": "min val_AQI_MAE"  // 监控指标
  }
}
```

---

## 📊 评估指标

模型使用以下指标评估预测性能：

| 指标 | 描述 | 计算时间段 |
|-----|------|----------|
| **AQI_MAE_112** | 平均绝对误差 | 1-12小时 |
| **AQI_MAE_1324** | 平均绝对误差 | 13-24小时 |
| **AQI_MAE_2548** | 平均绝对误差 | 25-48小时 |
| **AQI_RMSE_112** | 均方根误差 | 1-12小时 |
| **AQI_RMSE_1324** | 均方根误差 | 13-24小时 |
| **AQI_RMSE_2548** | 均方根误差 | 25-48小时 |
| **AQI_MAE** | 整体平均绝对误差 | 1-48小时 |
| **AQI_RMSE** | 整体均方根误差 | 1-48小时 |
| **Static_L1_Sparsity** | 因果矩阵稀疏度 | - |

---

## 📝 引用

如果您在研究中使用了 CausalAir，请引用：

```bibtex
@article{causalair2025,
  title={CausalAir: Causal Spatio-Temporal Air Quality Prediction Network},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下开源项目的启发和帮助：
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [TimesNet](https://github.com/thuml/Time-Series-Library)

---

## 🔄 更新日志

### v1.0.0 (2025-12-01)
- ✨ 初始版本发布
- 🎯 实现离散因果矩阵学习
- 🔥 双层因果建模机制
- ⚡ 温度退火策略
- 📊 完整的训练和评估流程

---

**⭐ 如果觉得这个项目有帮助，请给我们一个 Star！**
