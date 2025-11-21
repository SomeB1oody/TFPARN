# ASVspoof5 Deep Learning System

A Transformer-based deep learning system for detecting AI-generated synthetic speech in the ASVspoof5 challenge. This model distinguishes between genuine human speech (bonafide) and AI-generated synthetic speech (spoof) using a complete end-to-end architecture.

---

## Environment Setup

### Requirements

**Python Version:** 3.10 or higher

**Hardware Requirements:**
- **GPU (Recommended):** NVIDIA GPU with 8GB+ VRAM and CUDA 13.0
- **CPU:** 8-core processor (training on CPU is supported but significantly slower)
- **RAM:** 32GB+ system memory recommended
- **Storage:** ~100GB for datasets

---

## Dataset Preparation

### Dataset Structure

**Label Mapping:**
- `bonafide` → Genuine human speech (Label=1)
- `spoof` → AI-generated speech (Label=0)

### Data Download

Download ASVspoof datasets from:
- ASVspoof 2021: https://www.kaggle.com/datasets/mohammedabdeldayem/avsspoof-2021
- ASVspoof 2019: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
- ASVspoof 5: https://zenodo.org/records/14498691

---

## Training

### Basic Training

Edit the data paths in `main_train.py`:

```python
# In main_train.py, modify the DefaultArgs section:
args = DefaultArgs()
args.train_data_dir = "path/to/train/flac/"
args.train_protocol_dir = "path/to/train/protocol.tsv"
args.dev_data_dir = "path/to/dev/flac/"
args.dev_protocol_dir = "path/to/dev/protocol.tsv"
args.eval_data_dir = "path/to/eval/flac/"
args.eval_protocol_dir = "path/to/eval/protocol.tsv"
```

Run training:

```bash
python main_train.py
```

### Training Parameters

Key hyperparameters can be modified in `main_train.py`:

```python
@dataclass
class ModelArgs:
    # Training hyperparameters
    max_epochs: int = 80                    # Maximum training epochs
    batch_size: int = 96                    # Batch size (adjust based on GPU memory)
    learning_rate: float = 1e-4             # Initial learning rate
    weight_decay: float = 1e-2              # Weight decay for regularization

    # Model architecture
    d_model: int = 256                      # Model dimension
    num_layers: int = 6                     # Number of Transformer layers
    nhead: int = 8                          # Number of attention heads
    dropout: float = 0.3                    # Dropout rate

    # Audio processing
    n_mels: int = 128                       # Number of mel filterbanks
    duration_sec: float = 4.0               # Audio duration in seconds

    # Loss function
    loss_type: str = "focal"                # 'focal' or 'ce'
    focal_alpha: float = 0.1                # Focal loss alpha (weight for positive class)
    focal_gamma: float = 2.0                # Focal loss gamma (focusing parameter)

    # Augmentation
    use_rawboost: bool = True               # Enable RawBoost augmentation
    rawboost_prob: float = 0.5              # Probability of applying RawBoost

    # Test-Time Augmentation
    use_tta: bool = True                    # Enable TTA for validation/evaluation
    tta_num_crops: int = 5                  # Number of crops for TTA

    # Early stopping
    early_stopping_patience: int = 15       # Patience for early stopping

    # Model saving
    save_dir: str = "./final_nc/"           # Directory to save models
```

### Adjusting Batch Size for Different GPUs

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 8GB      | 64                     |
| 10GB     | 96                     |
| 12GB+    | 128                    |

To change batch size, modify in `main_train.py`:

```python
args.batch_size = 64  # For 8GB GPU
```

---

## Evaluation

### Basic Evaluation

Edit the model path and data paths in `read_and_evaluate.py`:

```python
# Model configuration
model_path = "./final_nc/best_model_eer_0.0567/best_model.pt"
config_path = "./final_nc/best_model_eer_0.0567/best_model.json"

# Dataset paths
datasets = [
    {
        "name": "Dev",
        "data_dir": "path/to/dev/flac/",
        "protocol_dir": "path/to/dev/protocol.tsv",
        "use_tta": True,
    },
    {
        "name": "Eval",
        "data_dir": "path/to/eval/flac/",
        "protocol_dir": "path/to/eval/protocol.tsv",
        "apply_calibration": True,
        "use_tta": True,
    }
]
```

Run evaluation:

```bash
python read_and_evaluate.py
```

### Evaluation Metrics

The system computes the following metrics:

- **EER (Equal Error Rate):** Point where false positive rate equals false negative rate. Lower is better.
- **minDCF (Minimum Detection Cost Function):** Weighted combination of error rates. Lower is better.
- **CLLR (Calibrated Log-Likelihood Ratio):** Measures calibration quality. Lower is better.
- **AUC-ROC:** Area under the ROC curve. Higher is better.
- **Accuracy, F1-Score:** Standard classification metrics.

---

## Model Configuration

### Architecture Overview

```
Raw Waveform → Log-Mel Spectrogram → Transformer Encoder → Pooling → Classification
```

**Key Features:**
- In-model mel spectrogram computation (no preprocessing needed)
- 6-layer Transformer encoder with 8 attention heads
- Flexible pooling strategies (mean/attention/top-k)
- End-to-end trainable

### Modifying Model Architecture

To change the model architecture, edit `SpeechClassifierArgs` in `main_train.py`:

```python
from model import SpeechClassifierArgs

# Example: Deeper model
args = SpeechClassifierArgs(
    d_model=256,           # Increase for wider model
    num_layers=8,          # Increase for deeper model
    nhead=8,
    dim_feedforward=1024,  # Increase for wider FFN
    dropout=0.3,
    pooling_method="mean"  # Options: "mean", "attention", "top-k"
)
```

### Pooling Methods

Three pooling strategies are available:

1. **Mean Pooling (Default):** Average all frame embeddings
   - Fast and memory-efficient
   - Good for most cases

2. **Attention Pooling:** Learned attention weights
   - Better performance but slower
   - Use when computational resources allow

3. **Top-k Pooling:** Select top-k frames by L2 norm
   - Focuses on most important frames
   - Requires tuning `top_k_ratio` parameter

To change pooling method:

```python
args.pooling_method = "attention"  # or "mean", "top-k"
args.top_k_ratio = 0.3  # Only for top-k pooling
```

### Data Augmentation

**RawBoost Augmentation:**
- Three augmentation algorithms (convolution, filtering, noise)
- Applied during training only
- Improves generalization

Configure in `main_train.py`:

```python
args.use_rawboost = True        # Enable/disable RawBoost
args.rawboost_prob = 0.5        # Probability of applying (0.0-1.0)
```

**Test-Time Augmentation (TTA):**
- Generates multiple crops per sample during inference
- Averages predictions for robustness
- Typically improves EER by 2-3%

Configure in `main_train.py`:

```python
args.use_tta = True             # Enable/disable TTA
args.tta_num_crops = 5          # Number of crops (3-7 recommended)
```

---

## Hyperparameter Tuning

To run multiple experiments with different hyperparameters:

```bash
python run_multiple_experiments.py
```

Edit the parameter grid in `run_multiple_experiments.py`:

```python
param_grid = {
    'focal_alpha': [0.05, 0.1, 0.2],
    'focal_gamma': [1.0, 2.0, 3.0],
    'pooling_method': ['mean', 'attention'],
    'learning_rate': [1e-4, 5e-5],
}
```

This will automatically run all combinations and save results to separate directories.

---

## References

- ASVspoof 5 Challenge: https://zenodo.org/records/14498691
- RawBoost Augmentation: https://arxiv.org/abs/2111.04433
- Focal Loss: https://arxiv.org/abs/1708.02002

---

## License

[MIT LICENSE](LICENSE)

---

## Contact

For questions or issues, please open an issue on GitHub or contact stanyin64@gmail.com