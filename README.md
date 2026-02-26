# TFPARN (Transformer-based Focal-Pairwise Attentive Ranking Network) for Anti-Spoofing

A Transformer solution for detecting AI-generated synthetic speech in the ASVspoof5 challenge. This model distinguishes between genuine human speech (bonafide) and AI-generated synthetic speech (spoof) using a complete end-to-end architecture.

---

## Environment Setup

### Requirements

**Python Version:** 3.10 or higher

**Dependencies:**
```
torch~=2.9.0+cu130
numpy~=2.1.2
scikit-learn~=1.7.1
tqdm~=4.67.1
torchaudio~=2.9.0
soundfile~=0.13.1
scipy~=1.15.3
```

Install via:
```bash
pip install -r requirements.txt
```

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
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """
    Complete configuration for training
    Includes: data processing, model architecture, and training hyperparameters
    """
    # Data paths
    train_data_dir: str = "N:/Dataset/ASV5/flac_T/"
    dev_data_dir: str = "N:/Dataset/ASV5/flac_D/"
    eval_data_dir: str = "N:/Dataset/ASV5/flac_E/"

    # Protocol file paths
    train_protocol_dir: str = "N:/Dataset/ASV5/ASVspoof5.train.tsv"
    dev_protocol_dir: str = "N:/Dataset/ASV5/ASVspoof5.dev.track_1.tsv"
    eval_protocol_dir: str = "N:/Dataset/ASV5/ASVspoof5.eval.track_1.tsv"

    # ...
```

Run training:

```bash
python main_train.py
```

### Training Parameters

Key hyperparameters can be modified in `main_train.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # ...
    
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

    # Early stopping
    early_stopping_patience: int = 15       # Patience for early stopping
    early_stopping_metric: str = "eer"      # Metric to monitor ('eer', 'f1_macro', 'accuracy')
    early_stopping_mode: str = "min"        # 'min' for eer, 'max' for f1/acc

    # Model saving
    save_dir: str = "./checkpoints/"        # Directory to save models
```

### Adjusting Batch Size for Different GPUs

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 8GB      | 64                     |
| 10GB     | 96                     |
| 12GB+    | 128                    |

To change batch size, modify in `main_train.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # ...
    
    batch_size = 64  # For 8GB GPU

    #...
```

---

## Evaluation

### Basic Evaluation

Edit the model path and data paths in `read_and_evaluate.py`:

```python
from dataclasses import dataclass, field
from typing import List
from read_and_evaluate import DatasetConfig

@dataclass
class EvaluationConfig:
    """
    Evaluation configuration parameters
    """
    # Model path
    model_path: str = "./checkpoints/best_model.pt"

    # Dataset configurations
    datasets: List[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig(
            name="Dev",
            data_dir="N:/Dataset/ASV5/flac_D/",
            protocol_dir="N:/Dataset/ASV5/ASVspoof5.dev.track_1.tsv",
        ),
        DatasetConfig(
            name="Eval",
            data_dir="N:/Dataset/ASV5/flac_E/",
            protocol_dir="N:/Dataset/ASV5/ASVspoof5.eval.track_1.tsv",
            apply_calibration=True,
        )
    ])
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
Raw Waveform -> Log-Mel Spectrogram -> Transformer Encoder -> Pooling -> Classification
```

**Key Features:**
- In-model mel spectrogram computation (no preprocessing needed)
- 6-layer Transformer encoder with 8 attention heads
- Flexible pooling strategies (mean/attention/top-k)
- End-to-end trainable

### Modifying Model Architecture

To change the model architecture, edit `SpeechClassifierArgs` in `main_train.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # ...
    
    # Model architecture
    n_mels: int = 160                       # Number of mel filterbanks
    n_fft: int = 1024                       # FFT window size
    hop_length: int = 160                   # Hop length for STFT
    d_model: int = 256                      # Model dimension
    nhead: int = 8                          # Number of attention heads
    num_layers: int = 6                     # Number of Transformer layers
    dim_feedforward: int = 1024             # Feedforward dimension
    model_dropout: float = 0.3              # Dropout rate
    activation: str = "relu"                # Activation function
    pooling_method: str = "mean"            # Options: "mean", "attention", "top-k"
    top_k_ratio: float = 0.3                # For top-k pooling: ratio of frames to keep

    # ...
```

### Pooling Methods

Three pooling strategies are available:

1. **Mean Pooling:** Average all frame embeddings
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
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # ...
    
    pooling_method = "attention"  # or "mean", "top-k"
    top_k_ratio = 0.3  # Only for top-k pooling

    # ...
```

### Data Augmentation

**RawBoost Augmentation:**
- Three augmentation algorithms (convolution, filtering, noise)
- Applied during training only
- Improves generalization

Configure in `main_train.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # ...

    use_rawboost: bool = True        # Enable/disable RawBoost

    # ...
```

---

## Multiple Experiments

To run multiple experiments with different parameters at once:

```bash
python run_multiple_experiments.py
```

Edit the parameter in `create_experiment_list` function of `run_multiple_experiments.py`:

```python
from typing import List
from main_train import ModelArgs

def create_experiment_list() -> List[ModelArgs]:
    """
    Define multiple experiments here
    Each experiment is a complete ModelArgs configuration

    Returns:
        List of ModelArgs configurations to run
    """
    experiments = []

    # Experiment 1: Focal Loss with Attention Pooling
    exp1 = ModelArgs()
    exp1.learning_rate = 1e-4
    exp1.weight_decay = 1e-2
    exp1.pooling_method = "attention"
    exp1.loss_type = "focal"
    exp1.enable_pairwise = True
    exp1.focal_alpha = 0.5
    exp1.focal_gamma = 2.0
    exp1.save_dir = "./final_TFPARN/focal_0.5_2.0_attention/"

    # Experiment 2: Focal Loss with Top-k Pooling
    exp2 = ModelArgs()
    exp2.learning_rate = 1e-4
    exp2.weight_decay = 1e-2
    exp2.pooling_method = "top-k"
    exp2.loss_type = "focal"
    exp2.enable_pairwise = True
    exp2.focal_alpha = 0.5
    exp2.focal_gamma = 2.0
    exp2.save_dir = "./final_TFPARN/focal_0.5_2.0_top-k/"

    # Experiment 3: Focal Loss with Mean Pooling
    exp3 = ModelArgs()
    exp3.learning_rate = 1e-4
    exp3.weight_decay = 1e-2
    exp3.pooling_method = "mean"
    exp3.loss_type = "focal"
    exp3.enable_pairwise = True
    exp3.focal_alpha = 0.5
    exp3.focal_gamma = 2.0
    exp3.save_dir = "./final_TFPARN/focal_0.5_2.0/"

    # More can be added here...

    experiments.append(exp1)
    experiments.append(exp2)
    experiments.append(exp3)
    # More can be added here...

    return experiments
```

This will automatically run all experiments in order and save the results in the specified directory.

---

## License

[MIT LICENSE](LICENSE)