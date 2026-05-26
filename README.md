# TFPARN (Transformer-based Focal-Pairwise Attentive Ranking Network) for Anti-Spoofing

A Transformer solution for detecting AI-generated synthetic speech in the ASVspoof5 challenge. The model distinguishes between genuine human speech (bonafide) and AI-generated synthetic speech (spoof) with a complete end-to-end architecture: the log-mel spectrogram is computed *inside* the model, a Transformer encoder processes it, and a classification head produces binary logits.

The name reflects the design: a **Transformer** backbone trained with **Focal** loss and a **Pairwise** ranking loss, aggregated with **Attentive** pooling to directly optimize **Ranking** metrics (EER / minDCF).

This repository also ships two re-implemented baselines (**AASIST** and **RawNet2**) and a **CSV analyzer** that turns the per-epoch logs into training-efficiency plots and statistics.

---

## Project Structure

```
TFPARN/
│
├── Model/                          # Main TFPARN model + the two baselines
│   ├── model.py                    # SpeechTransformerClassifier (in-model mel + Transformer)
│   ├── data_process.py             # Protocol parsing, ASV5Dataset, RawBoost, TTA, DataLoaders
│   ├── utils.py                    # Metrics, losses, calibration, checkpointing, early stopping
│   ├── main_train.py               # End-to-end training pipeline (ModelArgs config)
│   ├── read_and_evaluate.py        # Evaluate a saved checkpoint from its JSON config
│   ├── run_multiple_experiments.py # Run several training configs back-to-back
│   ├── requirements.txt            # Dependencies for training / evaluation (PyTorch + CUDA 13)
│   │
│   └── Baselines/
│       ├── AASIST/                 # AASIST baseline (SincConv + graph attention)
│       │   ├── model.py            # AASIST / AASISTArgs
│       │   ├── data_process.py
│       │   ├── main_train.py
│       │   ├── read_and_evaluate.py
│       │   └── utils.py
│       └── RawNet2/                # RawNet2 baseline (SincNet front-end + GRU)
│           ├── model.py            # RawNet2 / RawNet2Args
│           ├── data_process.py
│           ├── main_train.py
│           ├── read_and_evaluate.py
│           └── utils.py
│
├── CsvAnalyzer/                    # Training-efficiency analysis & plotting
│   ├── csv_analyzer.py             # Single-model report
│   ├── csv_compare.py              # Cross-model comparison
│   ├── csv_common.py               # Shared loading / interpolation / plotting / stats
│   ├── PLOTTING_METHOD.md          # Detailed description of the plotting method
│   └── requirements.txt            # Dependencies for plotting (matplotlib, SciencePlots, Jupyter)
│
├── Introduction_of_TFPARN.ipynb    # Full technical documentation notebook
├── README.md                       # This file
└── LICENSE
```

Each model directory (`Model/`, `Model/Baselines/AASIST/`, `Model/Baselines/RawNet2/`) is **self-contained** and uses flat, relative imports (`from data_process import ...`). Run the scripts from *inside* the directory that contains them, e.g. `cd Model && python main_train.py`.

---

## Environment Setup

**Python Version:** 3.10 or higher

There are two requirement files for two separate purposes:

| File | Purpose | Key packages |
|------|---------|--------------|
| `Model/requirements.txt` | Training & evaluation | `torch==2.11.0`, `torchaudio==2.11.0`, `numpy==2.4.4`, `scikit-learn==1.7.2`, `scipy==1.15.3`, `soundfile==0.13.1`, `tqdm==4.67.3` (plus the CUDA 13 `nvidia-*` wheels) |
| `CsvAnalyzer/requirements.txt` | Plotting / analysis | `matplotlib==3.10.8`, `SciencePlots==2.2.1`, `numpy==2.4.4`, Jupyter (`ipykernel`, `ipython`) |

Install the set you need:

```bash
# For training / evaluating models
pip install -r Model/requirements.txt

# For generating the training-efficiency plots
pip install -r CsvAnalyzer/requirements.txt
```

**Recommended Environment** (sized to run *everything* in this repo, including the AASIST baseline):
- **OS:** Linux
- **CPU:** 16-core processor or higher
- **RAM:** 128GB or more
- **Storage:** 500GB or more of free disk space
- **GPU:** A CUDA-capable NVIDIA GPU with 90GB+ VRAM (developed and tested on an **RTX Pro 6000**)

**Per-model VRAM:** TFPARN itself is light — it trains on a single 8GB+ GPU (≈8GB at the default `batch_size=64`), or even on CPU (much slower). The 90GB+ recommendation is driven by the **AASIST baseline**, which is very memory-hungry and needs ~80–90GB VRAM; TFPARN and RawNet2 are far more modest.

---

## Dataset Preparation

**Label Mapping:**
- `bonafide` → Genuine human speech (Label = 1)
- `spoof` → AI-generated speech (Label = 0)

**Protocol files** are the standard ASVspoof5 10-column whitespace-separated format:

```
speaker_id  file_name  gender  codec  codec_q  codec_seed  attack_tag  attack_label  KEY  tmp
```

The `KEY` field (`spoof` / `bonafide`) is the label. Audio is read directly from `.flac` files via `soundfile`.

**Data Download:**
- ASVspoof 5: https://zenodo.org/records/14498691
- ASVspoof 2021: https://www.kaggle.com/datasets/mohammedabdeldayem/avsspoof-2021
- ASVspoof 2019: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

---

## Training

### Basic Training

All paths in `ModelArgs` are **empty by default** and must be filled in. Edit `ModelArgs` at the top of `Model/main_train.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # Data paths (FLAC directories) — fill these in
    train_data_dir: str = "/path/to/ASV5/flac_T/"
    dev_data_dir: str   = "/path/to/ASV5/flac_D/"
    eval_data_dir: str  = "/path/to/ASV5/flac_E/"

    # Protocol file paths — fill these in
    train_protocol_dir: str = "/path/to/ASV5/ASVspoof5.train.tsv"
    dev_protocol_dir: str   = "/path/to/ASV5/ASVspoof5.dev.track_1.tsv"
    eval_protocol_dir: str  = "/path/to/ASV5/ASVspoof5.eval.track_1.tsv"

    # Where to save checkpoints — fill this in
    save_dir: str = "./checkpoints/"
    # ...
```

Then run from inside `Model/`:

```bash
cd Model
python main_train.py
```

### Training Parameters

The full configuration lives in the `ModelArgs` dataclass in `Model/main_train.py`. The defaults that define **TFPARN** are:

```python
@dataclass
class ModelArgs:
    # Audio processing
    sample_rate: int = 16000
    duration_sec: float = 4.0           # 4 s @ 16 kHz = 64,000 samples
    mono: bool = True
    normalize: bool = True

    # DataLoader
    batch_size: int = 64                # adjust to your GPU memory
    num_workers: int = 32
    prefetch_factor: int = 4

    # Augmentation
    use_rawboost: bool = True           # RawBoost (training only)
    rawboost_prob: float = 0.5
    use_tta: bool = True                # Test-Time Augmentation for dev/eval
    tta_num_crops: int = 5

    # Model architecture
    n_mels: int = 160                   # number of mel filterbanks
    n_fft: int = 1024                   # FFT window size
    hop_length: int = 160               # STFT hop length
    d_model: int = 256                  # model dimension
    nhead: int = 8                      # attention heads
    num_layers: int = 6                 # Transformer encoder layers
    dim_feedforward: int = 1024         # feed-forward dimension
    model_dropout: float = 0.3
    activation: str = "relu"            # "relu" or "gelu"
    pooling_method: str = "attention"   # "mean", "attention", or "top-k"
    top_k_ratio: float = 0.3            # only for top-k pooling

    # Training hyperparameters
    max_epochs: int = 100
    learning_rate: float = 0.667e-4
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"       # "adam" or "adamw"
    scheduler_type: str = "cosine"      # "cosine", "step", or "none"
    scheduler_warmup_epochs: int = 5

    # Loss function
    loss_type: str = "focal"            # "focal" or "ce"
    focal_alpha: float = 0.5            # positive-class weight (negative gets 1 - alpha)
    focal_gamma: float = 2.0            # focusing parameter

    # Pairwise ranking loss
    enable_pairwise: bool = True
    pairwise_margin: float = 1.0
    pairwise_weight: float = 0.3        # total loss = main_loss + weight * pairwise_loss

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "min_dcf"  # 'eer', 'min_dcf', 'f1_macro', 'accuracy', 'recall_macro', 'auc_roc'
    early_stopping_mode: str = "min"        # 'min' for eer/min_dcf, 'max' for the rest

    seed: int = 42
```

### Adjusting Batch Size for Different GPUs

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 8GB      | 64                     |
| 10GB     | 96                     |
| 12GB+    | 128                    |

Change `batch_size` in `ModelArgs`.

### Training Outputs

Training selects the best epoch by the `early_stopping_metric` on the **Dev** set (with TTA), reloads that checkpoint, and runs a final evaluation on Train/Dev/Eval. It writes, under `save_dir`:

```
save_dir/
├── best_model.pt                                   # rolling best checkpoint during training
└── best_model_<metric>_<value>/
    ├── best_model_<metric>_<value>.pt              # final weights + optimizer state
    ├── best_model_<metric>_<value>.json            # full config (data/model/train args) + metrics
    └── timing_records.csv                           # per-run time, memory, and metrics (Train/Dev/Eval)
```

The `timing_records.csv` is exactly the input consumed by the CSV analyzer (see below). Its `type` column is `1`=Train, `2`=Dev, `3`=final Eval.

---

## Evaluation

`Model/read_and_evaluate.py` re-loads a saved model **from its JSON config** (so the architecture is reconstructed exactly) and evaluates it on the Train/Dev/Eval sets. Edit `EvaluationConfig`:

```python
from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    # Required: the checkpoint and the JSON config saved during training
    json_path: str  = "./checkpoints/best_model_min_dcf_0.1234/best_model_min_dcf_0.1234.json"
    model_path: str = "./checkpoints/best_model_min_dcf_0.1234/best_model_min_dcf_0.1234.pt"

    # Dataset paths (can point to different data than training)
    train_data_dir: str = "/path/to/ASV5/flac_T/"
    dev_data_dir: str   = "/path/to/ASV5/flac_D/"
    eval_data_dir: str  = "/path/to/ASV5/flac_E/"

    train_protocol_dir: str = "/path/to/ASV5/ASVspoof5.train.tsv"
    dev_protocol_dir: str   = "/path/to/ASV5/ASVspoof5.dev.track_1.tsv"
    eval_protocol_dir: str  = "/path/to/ASV5/ASVspoof5.eval.track_1.tsv"

    use_tta: bool = True   # TTA is applied to dev/eval (train is always evaluated without TTA)
```

Run from inside `Model/`:

```bash
cd Model
python read_and_evaluate.py
```

### Evaluation Metrics

`compute_all_metrics` reports:

- **EER (Equal Error Rate):** point where the false-positive rate equals the false-negative rate. Lower is better.
- **minDCF (Minimum Detection Cost Function):** ASVspoof5 Track 1 normalized cost (`C_miss=1.0`, `C_fa=10.0`, `π_spf=0.05`). Lower is better.
- **actDCF (Actual DCF):** normalized cost at the Bayes-optimal threshold `τ = -log(β)`. Lower is better.
- **CLLR (Cost of Log-Likelihood Ratio):** calibration quality (computed during training and the calibrated evaluation path). Lower is better; 0 is perfect.
- **AUC-ROC:** area under the ROC curve. Higher is better.
- **Accuracy, F1 (macro), Recall (macro):** standard classification metrics.

`utils.py` additionally provides a **Platt calibration + prior-correction** pipeline (`evaluate_with_calibration`, `apply_platt_calibration`, `apply_prior_correction`) that fits a logistic regression on Dev and applies it to Eval.

---

## Model Configuration

### Architecture Overview

```
Raw Waveform [B, 1, 64000]
        │
        ▼  in-model log-mel spectrogram (STFT + mel filterbank, registered as buffers)
Log-Mel [B, T', n_mels]
        │  LayerNorm → Linear projection → Sinusoidal positional encoding
        ▼
6-layer Transformer Encoder (8 heads, d_model=256, dim_feedforward=1024)
        │
        ▼  Pooling: mean / attention / top-k  (TFPARN uses attention)
Pooled [B, d_model]
        │  2-layer MLP head
        ▼
Logits [B, 2]   →  logits[:, 0] = spoof,  logits[:, 1] = bonafide
```

**Key Features:**
- In-model mel spectrogram computation — no separate preprocessing step, identical at train and inference time.
- The mel filterbank and Hann window are registered as non-trainable buffers.
- Flexible pooling strategies (mean / attention / top-k).
- End-to-end trainable.

The model architecture is configured by `SpeechClassifierArgs` in `Model/model.py`; `main_train.py` populates it from the matching fields in `ModelArgs`.

### Pooling Methods

1. **Mean Pooling** — masked average of all frame embeddings. Fast and memory-efficient.
2. **Attention Pooling** — a learned attention head weights frames before aggregation (the TFPARN default).
3. **Top-k Pooling** — keeps the `top_k_ratio` fraction of frames with the largest L2 norm, then averages them.

Set `pooling_method` (and `top_k_ratio` for top-k) in `ModelArgs`.

### Loss Functions

- **Focal Loss** (`loss_type="focal"`): down-weights easy examples to handle the spoof/bonafide imbalance. `focal_alpha` is the bonafide weight; the spoof weight is `1 - focal_alpha`.
- **Cross-Entropy** (`loss_type="ce"`): standard alternative.
- **Pairwise Ranking Loss** (`enable_pairwise=True`): pushes bonafide scores above spoof scores by a margin, directly targeting EER/minDCF. Combined as `total = main_loss + pairwise_weight * pairwise_loss`.

### Data Augmentation

**RawBoost** (training only, applied with `rawboost_prob`) implements three algorithms:
1. Linear/nonlinear convolutive noise (random FIR + optional tanh distortion)
2. IIR filtering (random lowpass/highpass/bandpass Butterworth filter)
3. Stationary additive noise (random SNR 10–40 dB)

**Test-Time Augmentation (TTA):** for dev/eval, `tta_num_crops` overlapping crops are generated per utterance and the logits are averaged.

---

## Multiple Experiments

To run several training configurations back-to-back, edit the `create_experiment_list()` function in `Model/run_multiple_experiments.py` (each experiment is a full `ModelArgs`), then run:

```bash
cd Model
python run_multiple_experiments.py
```

```python
from typing import List
from main_train import ModelArgs

def create_experiment_list() -> List[ModelArgs]:
    experiments = []

    # Example: Focal + Pairwise with attention pooling
    exp1 = ModelArgs()
    exp1.learning_rate = 0.667e-4
    exp1.pooling_method = "attention"
    exp1.loss_type = "focal"
    exp1.enable_pairwise = True
    exp1.focal_alpha = 0.5
    exp1.focal_gamma = 2.0
    exp1.save_dir = "./experiments/focal_pairwise_attention/"

    experiments.append(exp1)
    # add more ModelArgs here ...
    return experiments
```

Each experiment is trained, evaluated, and saved exactly like `main_train.py`. A combined `experiments_summary_<timestamp>.json` and a comparison table are written at the end. Remember to give each experiment a **distinct `save_dir`**.

---

## Baseline Models

Two raw-waveform baselines are provided under `Model/Baselines/`. They share the same `data_process.py` pipeline as TFPARN (FLAC loading, 4 s fixed-length crop/repeat, RawBoost, TTA), the same metrics (`compute_eer`, `compute_min_dcf`, `compute_act_dcf`, `compute_cllr`), and the same training/evaluation flow — only the model architecture differs. Run each from inside its own directory.

### AASIST (`Model/Baselines/AASIST/`)

- **Model:** `AASIST` (config `AASISTArgs`). A SincConv front-end (`first_conv=128`), six 2D residual blocks (SELU), and homogeneous + heterogeneous **graph attention** layers (`gat_dims=[64, 32]`) with graph pooling (`pool_ratios=[0.5, 0.7, 0.5, 0.5]`).
- **Training defaults:** `batch_size=64`, `learning_rate=2.667e-4`, `max_epochs=100`, `loss_type="ce"`, `early_stopping_metric="min_dcf"`.
- ⚠️ **Very VRAM-hungry:** AASIST needs ~80–90GB of VRAM at `batch_size=64` (this is what drives the recommended-GPU spec above). Lower `batch_size` to fit a smaller GPU.

```bash
cd Model/Baselines/AASIST
# fill in the data/protocol/save paths in main_train.py, then:
python main_train.py
```

### RawNet2 (`Model/Baselines/RawNet2/`)

- **Model:** `RawNet2` (config `RawNet2Args`). A SincNet convolution front-end (`sinc_out_channels=20`, `sinc_kernel_size=1024`), six 1D residual blocks with per-block attention (LeakyReLU 0.3), and a 3-layer **GRU** (`gru_node=1024`, `nb_gru_layer=3`) followed by FC layers.
- **Training defaults:** `batch_size=64`, `learning_rate=5e-5`, `max_epochs=100`, `loss_type="ce"`, `early_stopping_metric="min_dcf"`.

```bash
cd Model/Baselines/RawNet2
# fill in the data/protocol/save paths in main_train.py, then:
python main_train.py
```

Both baselines produce the same checkpoint layout and `timing_records.csv` as the main model, so their logs feed straight into the CSV analyzer.

---

## Training-Efficiency Analysis

`CsvAnalyzer/` turns the `timing_records.csv` files emitted during training into time-vs-metric plots (EER, minDCF, CLLR, actDCF) and a `summary.md`. The x-axis is **cumulative Train+Dev time** (log scale), so models with very different run lengths can be compared fairly. See `CsvAnalyzer/PLOTTING_METHOD.md` for the full method (sampling, interpolation across seeds, best-point stars, statistics).

Inputs are given **per model as three seed CSVs** (seeds 42, 63, 2026); leave a seed `""` if it wasn't run.

```bash
pip install -r CsvAnalyzer/requirements.txt
cd CsvAnalyzer
```

- **Single-model report** — edit `MODEL_NAME`, `SEED_CSV_PATHS`, and `OUTPUT_DIR_PATH` in `csv_analyzer.py`:

  ```bash
  python csv_analyzer.py
  ```

- **Cross-model comparison** — edit the `MODEL_INPUTS` list (one `(name, {seed: path})` entry per model) and `OUTPUT_DIR_PATH` in `csv_compare.py`:

  ```bash
  python csv_compare.py
  ```

Each run writes 8 PDFs (`{EER,Min_DCF,Cllr,actDCF}_per_{Train,Dev}.pdf`) plus `summary.md` into `OUTPUT_DIR_PATH`.

---

## License

[MIT LICENSE](LICENSE)
