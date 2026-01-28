# CausalAir: Causal Spatio-Temporal Air Quality Prediction Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Introduction

**CausalAir (Causal Air Quality Prediction Network)** is a deep learning model for urban air quality forecasting that achieves interpretable spatio-temporal sequence prediction through learning discrete causal matrices.

### Core Features

🎯 **Discrete Causal Discovery**: Uses Hard Gumbel-Softmax sampling to generate discrete 0/1 causal matrices for interpretable causal relationship learning

🔥 **Dual-Layer Causal Modeling**: 
- Station-level causal matrix [N, N]: Captures spatial dependencies between stations
- Variable-level causal matrix [N, 7, 56]: Models causal relationships between air quality features and meteorological features

⚡ **Efficient Gradient Propagation**: Employs Straight-Through Estimator (STE) for differentiable optimization of discrete sampling

📊 **Strong Interpretability**: Learned causal matrices clearly reveal causal dependencies between variables

🎚️ **Temperature Annealing Strategy**: Supports linear and exponential decay temperature scheduling to gradually enhance discretization during training

---

## 🏗️ Model Architecture

### Overall Architecture

```
Input: 
  - Historical air quality data [B, N, L, 7]
  - Historical meteorological data [B, N, L, 56]
  - Future meteorological forecast [B, N, P, 56]

        ↓
    
[Embedding Layer] → Historical Representation + Future Representation
        
        ↓
        
[Causal Matrix Generation]
  ├─ Station-level Causal Matrix [N, N] (Gumbel-Softmax)
  └─ Variable-level Causal Matrix [N, 7, 56] (Gumbel-Softmax)
        
        ↓
        
[CausalAir Encoder] (L Layers)
  ├─ Station-wise Self-Attention (with causal mask)
  ├─ Variable-wise Self-Attention
  ├─ Meteorological Feature Update (with variable-level causal mask)
  ├─ Historical-Future Cross-Attention
  └─ Feed-Forward Network
        
        ↓
        
[Prediction Heads]
  ├─ Initial Prediction (from historical representation)
  ├─ Final Prediction (from future representation)
  └─ Reconstruction Loss (historical reconstruction)

Output: 48-hour future air quality prediction [B, N, 48, 7]
```

### Key Components

#### 1. Causal Matrix Generation

```python
# Station-level causal matrix
theta: [N, N, 2] → Gumbel-Softmax → causal_matrix: [N, N]

# Variable-level causal matrix  
station_var_theta: [N, 7, 56, 2] → Gumbel-Softmax → var_causal_matrix: [N, 7, 56]
```

#### 2. Backdoor Replacement Mechanism

For attention computation with causal matrix:

```
O = (A ⊙ C) @ V + (A ⊙ (1-C)) @ Z
```

Where:
- `A`: Attention weight matrix
- `C`: Causal matrix (0 or 1)
- `V`: Original value matrix
- `Z`: Noise matrix

#### 3. Temperature Annealing

```python
# Exponential decay
τ(t) = τ_final + (τ_initial - τ_final) * exp(-5 * progress)

# Linear decay
τ(t) = τ_initial - (τ_initial - τ_final) * progress
```

---

## 📂 Project Structure

```
CausalAir/
├── base/                      # Base class definitions
│   ├── base_data_loader.py   # Data loader base class
│   ├── base_model.py          # Model base class
│   └── base_trainer.py        # Trainer base class
├── config/                    # Configuration files
│   └── causalair.json        # Model configuration
├── data_loader/              # Data loading modules
│   ├── sts_loader.py         # Spatio-temporal sequence data loader
│   └── dataset.py            # Dataset definition
├── layers/                   # Network layer definitions
│   ├── CausalAir_EncDec.py   # CausalAir encoder layers
│   ├── Embed.py              # Embedding layers
│   ├── SelfAttention_Family.py  # Attention mechanism family
│   └── ...
├── model/                    # Model definitions
│   └── CausalAir.py         # CausalAir main model
├── trainer/                  # Training logic
│   └── trainer.py           # Trainer implementation
├── evaluation/              # Evaluation metrics
│   └── metric.py           # Evaluation metrics
├── logger/                  # Logging module
├── utils/                   # Utility functions
├── train.py                 # Training script
├── test.py                  # Testing script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.1
- CUDA >= 10.2 (GPU training recommended)

### Installation

1. **Clone the project**
```bash
cd /mnt/hyin/workspace/
# Project exists in CausalAir directory
```

2. **Install dependencies**
```bash
cd CausalAir
pip install -r requirements.txt
```

3. **Prepare data**

Data should be placed at:
```
/mnt/hyin/Datasets/china_stations_data/aq_level_mete/
├── train_data.pkl   # Training data
├── val_data.pkl     # Validation data
└── test_data.pkl    # Test data
```

Data format:
- `aq_train_data`: [B, T, N, 7] - Air quality data (7 features)
- `mete_train_data`: [B, T, N, 56] - Meteorological data (56 features)
- `AQStation_coordinate`: [N, 2] - Station coordinates

### Training

```bash
# Train with default configuration
python train.py --config config/causalair.json

# Multi-GPU training
python train.py --config config/causalair.json --device 0,1

# Resume from checkpoint
python train.py --config config/causalair.json --resume checkpoints/path/to/checkpoint.pth
```

### Testing

```bash
# Test model
python test.py --resume checkpoints/causalair_causal_air_quality_prediction/train/XXXXXX/model_best.pth

# Test and save predictions
python test.py --resume checkpoints/path/to/model_best.pth --save_output
```

### Monitoring

Monitor training with TensorBoard:

```bash
tensorboard --logdir=checkpoints/causalair_causal_air_quality_prediction/train/
```

---

## ⚙️ Configuration

Main configuration parameters (`config/causalair.json`):

### Model Parameters

```json
{
  "seq_len": 48,           // Historical sequence length (hours)
  "pred_len": 48,          // Prediction sequence length (hours)
  "d_model": 256,          // Model dimension
  "n_heads": 8,            // Number of attention heads
  "e_layers": 3,           // Number of encoder layers
  "d_ff": 1024,            // Feed-forward network dimension
  "dropout": 0.1,          // Dropout rate
  "n_station": 1628,       // Number of stations
  "gat_node_features": 7,  // Air quality features
  "mete_features": 56      // Meteorological features
}
```

### Causal Learning Parameters

```json
{
  "sparsity_weight": 5,    // Sparsity regularization weight
  "backdoor_attention": {
    "enabled": true,       // Enable backdoor attention
    "noise_type": "uniform",  // Noise type
    "noise_std": 0.1,      // Noise standard deviation
    "temperature": 1.0     // Initial temperature
  }
}
```

### Temperature Scheduler

```json
{
  "temperature_scheduler": {
    "type": "ExponentialDecay",  // Decay type: ExponentialDecay/LinearDecay
    "args": {
      "initial_temp": 1.0,       // Initial temperature
      "final_temp": 0.1,         // Final temperature
      "decay_epochs": 50,        // Decay epochs
      "decay_type": "exponential"  // Decay method
    }
  }
}
```

### Training Parameters

```json
{
  "optimizer": {
    "type": "Adam",
    "args": {
      "lr": 0.0001,        // Learning rate
      "weight_decay": 0,   // Weight decay
      "amsgrad": true
    }
  },
  "trainer": {
    "epochs": 100,         // Training epochs
    "save_period": 10,     // Save period
    "early_stop": 50,      // Early stopping patience
    "monitor": "min val_AQI_MAE"  // Monitoring metric
  }
}
```

---

## 📊 Evaluation Metrics

The model uses the following metrics to evaluate prediction performance:

| Metric | Description | Time Period |
|-----|------|----------|
| **AQI_MAE_112** | Mean Absolute Error | 1-12 hours |
| **AQI_MAE_1324** | Mean Absolute Error | 13-24 hours |
| **AQI_MAE_2548** | Mean Absolute Error | 25-48 hours |
| **AQI_RMSE_112** | Root Mean Square Error | 1-12 hours |
| **AQI_RMSE_1324** | Root Mean Square Error | 13-24 hours |
| **AQI_RMSE_2548** | Root Mean Square Error | 25-48 hours |
| **AQI_MAE** | Overall Mean Absolute Error | 1-48 hours |
| **AQI_RMSE** | Overall Root Mean Square Error | 1-48 hours |
| **Static_L1_Sparsity** | Causal Matrix Sparsity | - |

---

## 📝 Citation

If you use CausalAir in your research, please cite:

```bibtex
@article{causalair2025,
  title={CausalAir: Causal Spatio-Temporal Air Quality Prediction Network},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Thanks to the following open-source projects for inspiration and support:
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [TimesNet](https://github.com/thuml/Time-Series-Library)

---

## 🔄 Changelog

### v1.0.0 (2025-12-01)
- ✨ Initial release
- 🎯 Discrete causal matrix learning
- 🔥 Dual-layer causal modeling mechanism
- ⚡ Temperature annealing strategy
- 📊 Complete training and evaluation pipeline

---

**⭐ If you find this project helpful, please give us a Star!**
