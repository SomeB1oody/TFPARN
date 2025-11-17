"""
ASVspoof5 Data Loading Module
Provides data pipeline for ASVspoof 5 competition: protocol parsing, audio loading, fixed-length processing, DataLoader construction

Label mapping: spoof(AI-generated)=0, bonafide(genuine human)=1
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DefaultArgs:
    """
    Default arguments for data loading
    """
    # Data paths
    train_data_dir: str = "N:/Dataset/ASV5/flac_T/"
    dev_data_dir: str = "N:/Dataset/ASV5/flac_D/"
    eval_data_dir: str = "N:/Dataset/ASV5/flac_E/"

    # Protocol file paths
    train_protocol_dir: str = "N:/Dataset/ASV5/ASVspoof5.train.tsv"
    dev_protocol_dir: str = "N:/Dataset/ASV5/ASVspoof5.dev.track_1.tsv"
    eval_protocol_dir: str = "N:/Dataset/ASV5/ASVspoof5.eval.track_1.tsv"

    # Audio processing parameters
    sample_rate: int = 16000
    duration_sec: float = 4.0
    mono: bool = True
    normalize: bool = True  # Normalize to [-1, 1]

    # DataLoader parameters
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    train_shuffle: bool = True

    # Misc
    seed: int = 42

    # RawBoost augmentation parameters
    use_rawboost: bool = True  # Enable RawBoost data augmentation for training
    rawboost_prob: float = 0.5  # Probability of applying RawBoost

    # TTA parameters
    use_tta: bool = True  # Enable Test-Time Augmentation for dev/eval
    tta_num_crops: int = 5  # Number of crops per sample for TTA


def get_default_args() -> DefaultArgs:
    """
    Return default arguments instance
    """
    return DefaultArgs()


# ============================================================================
# RawBoost Data Augmentation
# ============================================================================

class RawBoost:
    """
    RawBoost data augmentation for anti-spoofing
    Reference: "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing"

    Applies various augmentations:
    - Algorithm 1: Linear/Nonlinear convolutive noise
    - Algorithm 2: IIR filtering (lowpass/highpass/bandpass)
    - Algorithm 3: Stationary additive noise
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor, algo: int = None) -> torch.Tensor:
        """
        Apply RawBoost augmentation

        Args:
            waveform: Input waveform [C, T]
            algo: Algorithm to use (1, 2, or 3). If None, randomly select

        Returns:
            Augmented waveform [C, T]
        """
        if algo is None:
            algo = np.random.choice([1, 2, 3])

        # Convert to numpy for processing
        x = waveform.squeeze(0).numpy()  # [T]

        if algo == 1:
            x = self._algo1_linear_nonlinear_convolution(x)
        elif algo == 2:
            x = self._algo2_iir_filtering(x)
        elif algo == 3:
            x = self._algo3_stationary_additive_noise(x)

        # Convert back to torch and add channel dimension
        return torch.from_numpy(x).unsqueeze(0).float()

    def _algo1_linear_nonlinear_convolution(self, x: np.ndarray) -> np.ndarray:
        """
        Algorithm 1: Apply linear/nonlinear convolutive noise
        """
        # Generate random FIR filter
        N_fir = np.random.randint(5, 15)
        h = np.random.randn(N_fir)
        h = h / np.sum(np.abs(h))  # Normalize

        # Apply convolution
        x_conv = signal.convolve(x, h, mode='same')

        # Random nonlinear distortion (tanh)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.1, 0.5)
            x_conv = np.tanh(alpha * x_conv)

        return x_conv.astype(np.float32)

    def _algo2_iir_filtering(self, x: np.ndarray) -> np.ndarray:
        """
        Algorithm 2: Apply IIR filtering (lowpass/highpass/bandpass)
        """
        # Randomly select filter type
        filter_type = np.random.choice(['lowpass', 'highpass', 'bandpass'])

        # Design filter
        if filter_type == 'lowpass':
            cutoff = np.random.uniform(1000, 4000)  # Hz
            b, a = signal.butter(4, cutoff / (self.sample_rate / 2), btype='low')
        elif filter_type == 'highpass':
            cutoff = np.random.uniform(200, 1000)  # Hz
            b, a = signal.butter(4, cutoff / (self.sample_rate / 2), btype='high')
        else:  # bandpass
            low = np.random.uniform(200, 1000)
            high = np.random.uniform(3000, 6000)
            b, a = signal.butter(4, [low / (self.sample_rate / 2), high / (self.sample_rate / 2)], btype='band')

        # Apply filter
        x_filtered = signal.filtfilt(b, a, x)

        return x_filtered.astype(np.float32)

    def _algo3_stationary_additive_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Algorithm 3: Add stationary noise with random SNR
        """
        # Generate noise
        noise = np.random.randn(len(x))

        # Random SNR between 10-40 dB
        snr_db = np.random.uniform(10, 40)

        # Calculate noise power
        signal_power = np.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Scale noise
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))

        # Add noise
        x_noisy = x + noise

        return x_noisy.astype(np.float32)


# ============================================================================
# Protocol Parsing
# ============================================================================

def read_protocol(protocol_path: str) -> List[Dict[str, Any]]:
    """
    Read and parse ASVspoof5 protocol file

    Args:
        protocol_path: Path to protocol file

    Returns:
        List of dictionaries containing parsed protocol information
        Each dict contains:
        - flac_file: FLAC filename (with .flac extension)
        - speaker_id: Speaker ID
        - gender: Gender (F/M)
        - codec: Codec
        - codec_q: Codec quality
        - codec_seed: Codec seed
        - attack_tag: Attack tag
        - attack_label: Attack label
        - label: 0=spoof(fake), 1=bonafide(genuine)
    """
    print(f"\n{'='*80}")
    print(f"[Protocol Parsing] Reading: {protocol_path}")
    print(f"{'='*80}")

    if not os.path.exists(protocol_path):
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_path}\n"
            f"Fix: Please check if the path is correct"
        )

    items = []
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 10:
                raise ValueError(
                    f"Invalid protocol format at line {line_num}:\n"
                    f"  Expected 10 columns, got {len(parts)}\n"
                    f"  Line content: {line}\n"
                    f"  File: {protocol_path}"
                )

            speaker_id, flac_name, gender, codec, codec_q, codec_seed, \
                attack_tag, attack_label, key, tmp = parts

            # Validate KEY field
            if key not in ['spoof', 'bonafide']:
                raise ValueError(
                    f"Invalid KEY field at line {line_num}:\n"
                    f"  Expected 'spoof' or 'bonafide'\n"
                    f"  Got: {key}\n"
                    f"  File: {protocol_path}"
                )

            # Add .flac extension uniformly
            if not flac_name.endswith('.flac'):
                flac_name = f"{flac_name}.flac"

            # Convert label: spoof=0(fake), bonafide=1(genuine)
            label = 1 if key == 'bonafide' else 0

            item = {
                'flac_file': flac_name,
                'speaker_id': speaker_id,
                'gender': gender,
                'codec': codec,
                'codec_q': codec_q,
                'codec_seed': codec_seed,
                'attack_tag': attack_tag,
                'attack_label': attack_label,
                'label': label,
            }
            items.append(item)

    print(f"✓ Successfully parsed {len(items)} entries")

    # Statistics
    bonafide_count = sum(1 for item in items if item['label'] == 1)
    spoof_count = len(items) - bonafide_count
    print(f"  - Bonafide (genuine human): {bonafide_count} ({100*bonafide_count/len(items):.2f}%)")
    print(f"  - Spoof (AI-generated): {spoof_count} ({100*spoof_count/len(items):.2f}%)")

    return items


# ============================================================================
# Dataset Class
# ============================================================================

class ASV5Dataset(Dataset):
    """
    ASVspoof5 Dataset

    Loads audio files, applies fixed-length processing using repetition strategy
    Returns dictionary with waveform, label, and metadata

    Label mapping: spoof=0, bonafide=1
    """

    def __init__(
        self,
        data_dir: str,
        items: List[Dict[str, Any]],
        sample_rate: int,
        duration_sec: float,
        mono: bool,
        normalize: bool,
        seed: int,
        mode: str,
        use_rawboost: bool = False,
        rawboost_prob: float = 0.5
    ):
        """
        Args:
            data_dir: Directory containing FLAC audio files
            items: List of parsed protocol items
            sample_rate: Target sample rate
            duration_sec: Fixed duration in seconds
            mono: Convert to mono
            normalize: Normalize to [-1, 1]
            seed: Random seed
            mode: Mode ("train"/"dev"/"eval")
            use_rawboost: Enable RawBoost augmentation (only for training)
            rawboost_prob: Probability of applying RawBoost
        """
        print(f"\n[Dataset Init] Initializing ASV5Dataset")
        print(f"  - Mode: {mode}")
        print(f"  - Data directory: {data_dir}")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Duration: {duration_sec} seconds ({int(sample_rate * duration_sec)} samples)")
        print(f"  - Number of items: {len(items)}")

        self.data_dir = Path(data_dir)
        self.items = items
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_length = int(sample_rate * duration_sec)
        self.mono = mono
        self.normalize = normalize
        self.seed = seed
        self.mode = mode
        self.use_rawboost = use_rawboost and mode == "train"  # Only use in training
        self.rawboost_prob = rawboost_prob

        # Random generator for training mode
        if mode == "train":
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None

        # Initialize RawBoost augmentation
        if self.use_rawboost:
            self.rawboost = RawBoost(sample_rate=sample_rate)
            print(f"  - RawBoost augmentation: ENABLED (prob={rawboost_prob})")
        else:
            self.rawboost = None

        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Fix: Please check if the path is correct"
            )

        print(f"✓ Dataset initialized successfully")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Load and process a single sample

        Returns:
            Dict with keys: waveform, length, label, metadata
        """
        item = self.items[index]
        filename = item['flac_file']
        label = item['label']
        audio_path = self.data_dir / filename

        try:
            # Load audio using soundfile (avoids Windows FFmpeg dependency issues)
            audio_data, sr = sf.read(str(audio_path), dtype='float32')

            # Convert to torch tensor and add channel dimension [T] -> [C, T]
            if audio_data.ndim == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data).T

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)

            # Convert to mono
            if self.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize to [-1, 1]
            if self.normalize:
                max_val = torch.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val

            # Apply fixed-length processing with repetition strategy
            waveform = self._apply_fixed_length(waveform)

            # Apply RawBoost augmentation (only for training mode)
            if self.use_rawboost and np.random.rand() < self.rawboost_prob:
                waveform = self.rawboost(waveform)

            return {
                "waveform": waveform,
                "length": waveform.shape[-1],
                "label": label,
                "speaker_id": item['speaker_id'],
                "attack_label": item['attack_label'],
                "audio_path": str(audio_path)
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio:\n"
                f"  Path: {audio_path}\n"
                f"  Error: {str(e)}\n"
                f"  Suggestion: Check if file exists, permissions, and format"
            ) from e

    def _apply_fixed_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Fixed-length processing: crop or repeat-concatenate to target length

        Strategy:
        - If audio is longer than target: crop (random for train, center for dev/eval)
        - If audio is shorter than target: repeat-concatenate until reaching target length

        Args:
            waveform: [C, T]

        Returns:
            Fixed-length waveform [C, target_length]
        """
        C, T = waveform.shape
        target_T = self.target_length

        if T == target_T:
            return waveform
        elif T > target_T:
            # Crop: random crop for training, center crop for dev/eval
            if self.mode == "train":
                # Random crop
                max_start = T - target_T
                start = torch.randint(0, max_start + 1, (1,), generator=self.generator).item()
            else:
                # Center crop
                start = (T - target_T) // 2
            return waveform[:, start:start + target_T]
        else:
            # Repeat-concatenate: repeat audio until reaching target length
            num_repeats = (target_T // T) + 1  # Calculate how many times to repeat
            repeated = waveform.repeat(1, num_repeats)  # Repeat along time axis

            # Crop to exact target length
            if self.mode == "train":
                # Random crop from repeated audio
                max_start = repeated.shape[1] - target_T
                start = torch.randint(0, max_start + 1, (1,), generator=self.generator).item()
            else:
                # Center crop from repeated audio
                start = (repeated.shape[1] - target_T) // 2

            return repeated[:, start:start + target_T]

    def generate_tta_crops(self, waveform: torch.Tensor, num_crops: int = 5) -> torch.Tensor:
        """
        Generate multiple crops for Test-Time Augmentation

        Strategy:
        - If audio is longer than target: generate overlapping sliding windows (50% overlap)
        - If audio is shorter than target: repeat with different random starting points

        Args:
            waveform: [C, T]
            num_crops: Number of crops to generate (default 5)

        Returns:
            Crops: [num_crops, C, target_length]
        """
        C, T = waveform.shape
        target_T = self.target_length
        crops = []

        if T >= target_T:
            # Longer than target: sliding windows with 50% overlap
            stride = target_T // 2
            max_start = T - target_T

            if max_start == 0:
                # Exact length: just use it multiple times (no variance)
                for _ in range(num_crops):
                    crops.append(waveform[:, :target_T])
            else:
                # Generate evenly spaced starting points
                step = max(1, max_start // (num_crops - 1))
                starts = [min(i * step, max_start) for i in range(num_crops)]

                for start in starts:
                    crops.append(waveform[:, start:start + target_T])
        else:
            # Shorter than target: repeat and use different starting points
            num_repeats = (target_T // T) + 1
            repeated = waveform.repeat(1, num_repeats)  # [C, repeated_T]

            max_start = repeated.shape[1] - target_T
            if max_start == 0:
                # Even after repeating, exact length
                for _ in range(num_crops):
                    crops.append(repeated[:, :target_T])
            else:
                # Generate evenly spaced starting points
                step = max(1, max_start // (num_crops - 1))
                starts = [min(i * step, max_start) for i in range(num_crops)]

                for start in starts:
                    crops.append(repeated[:, start:start + target_T])

        # Stack crops: [num_crops, C, target_T]
        return torch.stack(crops, dim=0)


# ============================================================================
# TTA Dataset Wrapper
# ============================================================================

class TTADataset(torch.utils.data.Dataset):
    """
    Test-Time Augmentation Dataset Wrapper

    Wraps an existing ASV5Dataset and generates multiple crops per sample
    for variance reduction during inference.
    """

    def __init__(self, base_dataset: ASV5Dataset, num_crops: int = 5):
        """
        Args:
            base_dataset: ASV5Dataset instance
            num_crops: Number of crops to generate per sample
        """
        self.base_dataset = base_dataset
        self.num_crops = num_crops
        print(f"\n[TTA Dataset] Wrapping dataset with {num_crops} crops per sample")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get item with TTA crops

        Returns:
            Dict with keys:
                - waveforms: [num_crops, C, T] multiple crops
                - length: int (all crops have same length)
                - label: int
                - speaker_id: str
                - attack_label: str
                - audio_path: str
        """
        # Get base item (with single center crop)
        item = self.base_dataset.items[index]
        filename = item['flac_file']
        label = item['label']
        audio_path = self.base_dataset.data_dir / filename

        try:
            # Load and preprocess audio (same as base dataset)
            import soundfile as sf
            import torchaudio

            audio_data, sr = sf.read(str(audio_path), dtype='float32')

            if audio_data.ndim == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data).T

            if sr != self.base_dataset.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.base_dataset.sample_rate
                )
                waveform = resampler(waveform)

            if self.base_dataset.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if self.base_dataset.normalize:
                max_val = torch.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val

            # Generate multiple crops using TTA
            waveforms = self.base_dataset.generate_tta_crops(waveform, self.num_crops)

            return {
                "waveforms": waveforms,  # [num_crops, C, T]
                "length": waveforms.shape[-1],
                "label": label,
                "speaker_id": item['speaker_id'],
                "attack_label": item['attack_label'],
                "audio_path": str(audio_path)
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio for TTA:\n"
                f"  Path: {audio_path}\n"
                f"  Error: {str(e)}"
            ) from e


def collate_fn_tta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for TTA batches

    Args:
        batch: List of dicts from TTADataset.__getitem__

    Returns:
        Dict with:
            waveforms: FloatTensor [B, num_crops, C, T]
            lengths: LongTensor [B]
            labels: LongTensor [B]
            speaker_ids: List[str]
            attack_labels: List[str]
            audio_paths: List[str]
    """
    waveforms = torch.stack([item["waveforms"] for item in batch])  # [B, num_crops, C, T]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    speaker_ids = [item["speaker_id"] for item in batch]
    attack_labels = [item["attack_label"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    return {
        "waveforms": waveforms,
        "lengths": lengths,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "attack_labels": attack_labels,
        "audio_paths": audio_paths
    }


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch and stack tensors

    Args:
        batch: List of dicts from __getitem__

    Returns:
        Dict with:
            waveforms: FloatTensor [B, C, T]
            lengths: LongTensor [B]
            labels: LongTensor [B]
            speaker_ids: List[str]
            attack_labels: List[str]
            audio_paths: List[str]
    """
    waveforms = torch.stack([item["waveform"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    speaker_ids = [item["speaker_id"] for item in batch]
    attack_labels = [item["attack_label"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    return {
        "waveforms": waveforms,
        "lengths": lengths,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "attack_labels": attack_labels,
        "audio_paths": audio_paths
    }


# ============================================================================
# Class Weights Calculation
# ============================================================================

def get_class_weights(items: List[Dict[str, Any]]) -> torch.Tensor:
    """
    Calculate class weights for loss function
    Weight calculation: weight[i] = max_count / count[i]

    Args:
        items: List of protocol items

    Returns:
        FloatTensor [2]: Class weights [spoof_weight, bonafide_weight]
    """
    print(f"\n[Class Weights] Calculating class weights for loss function")

    labels = [item['label'] for item in items]
    spoof_count = sum(1 for label in labels if label == 0)
    bonafide_count = sum(1 for label in labels if label == 1)

    if spoof_count == 0 or bonafide_count == 0:
        print(f"  Warning: One class has zero samples, using uniform weights")
        return torch.ones(2, dtype=torch.float32)

    majority_count = max(spoof_count, bonafide_count)
    weight_spoof = majority_count / spoof_count
    weight_bonafide = majority_count / bonafide_count

    weights = torch.tensor([weight_spoof, weight_bonafide], dtype=torch.float32)

    print(f"  - Spoof count: {spoof_count}, weight: {weight_spoof:.4f}")
    print(f"  - Bonafide count: {bonafide_count}, weight: {weight_bonafide:.4f}")

    return weights


# ============================================================================
# Main Function: Build DataLoaders
# ============================================================================

def make_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, dev, eval DataLoaders

    Args:
        args: Argument object containing all fields in DefaultArgs

    Returns:
        (train_loader, dev_loader, eval_loader)

    Raises:
        FileNotFoundError: Protocol file or audio directory not found
        ValueError: Protocol format error
    """
    print("\n" + "="*80)
    print("STARTING TO BUILD DATALOADERS")
    print("="*80)

    # Parse protocols
    print(f"\n[Step 1/3] Parsing protocol files")
    print(f"  - Train protocol: {args.train_protocol_dir}")
    train_items = read_protocol(args.train_protocol_dir)

    print(f"  - Dev protocol: {args.dev_protocol_dir}")
    dev_items = read_protocol(args.dev_protocol_dir)

    print(f"  - Eval protocol: {args.eval_protocol_dir}")
    eval_items = read_protocol(args.eval_protocol_dir)

    # Build datasets
    print(f"\n[Step 2/3] Building datasets")

    train_dataset = ASV5Dataset(
        data_dir=args.train_data_dir,
        items=train_items,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        mono=args.mono,
        normalize=args.normalize,
        seed=args.seed,
        mode="train",
        use_rawboost=args.use_rawboost,
        rawboost_prob=args.rawboost_prob
    )

    dev_base_dataset = ASV5Dataset(
        data_dir=args.dev_data_dir,
        items=dev_items,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        mono=args.mono,
        normalize=args.normalize,
        seed=args.seed,
        mode="dev",
        use_rawboost=False,  # No augmentation for dev
        rawboost_prob=0.0
    )

    eval_base_dataset = ASV5Dataset(
        data_dir=args.eval_data_dir,
        items=eval_items,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        mono=args.mono,
        normalize=args.normalize,
        seed=args.seed,
        mode="eval",
        use_rawboost=False,  # No augmentation for eval
        rawboost_prob=0.0
    )

    # Wrap dev/eval with TTA if enabled
    if args.use_tta:
        print(f"\n[TTA] Test-Time Augmentation ENABLED")
        print(f"  - Number of crops per sample: {args.tta_num_crops}")
        dev_dataset = TTADataset(dev_base_dataset, num_crops=args.tta_num_crops)
        eval_dataset = TTADataset(eval_base_dataset, num_crops=args.tta_num_crops)
        dev_collate_fn = collate_fn_tta
        eval_collate_fn = collate_fn_tta
    else:
        print(f"\n[TTA] Test-Time Augmentation DISABLED")
        dev_dataset = dev_base_dataset
        eval_dataset = eval_base_dataset
        dev_collate_fn = collate_fn
        eval_collate_fn = collate_fn

    # Build DataLoaders
    print(f"\n[Step 3/3] Building DataLoaders")
    print(f"  Configuration:")
    print(f"    - Batch size: {args.batch_size}")
    print(f"    - Num workers: {args.num_workers}")
    print(f"    - Prefetch factor: {args.prefetch_factor}")
    print(f"    - Pin memory: {args.pin_memory}")
    print(f"    - Persistent workers: {args.persistent_workers}")
    print(f"    - Train shuffle: {args.train_shuffle}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.train_shuffle,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=collate_fn
    )
    print(f"  ✓ Train DataLoader ready: {len(train_loader)} batches")

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=dev_collate_fn
    )
    print(f"  ✓ Dev DataLoader ready: {len(dev_loader)} batches")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=eval_collate_fn
    )
    print(f"  ✓ Eval DataLoader ready: {len(eval_loader)} batches")

    print("\n" + "="*80)
    print("DATALOADER CONSTRUCTION COMPLETE")
    print("="*80)
    print(f"Summary:")
    print(f"  - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  - Dev: {len(dev_dataset)} samples, {len(dev_loader)} batches")
    print(f"  - Eval: {len(eval_dataset)} samples, {len(eval_loader)} batches")
    if args.use_tta:
        print(f"  - TTA: {args.tta_num_crops} crops per sample")
    print("="*80 + "\n")

    return train_loader, dev_loader, eval_loader


# ============================================================================
# Test/Demo Function
# ============================================================================

if __name__ == "__main__":
    """
    Quick test of the data loading pipeline
    """
    print("Testing ASVspoof5 data loading pipeline...")

    # Get default args
    args = get_default_args()

    # Override for quick test (optional)
    args.batch_size = 4
    args.num_workers = 0  # Set to 0 for debugging

    try:
        # Build loaders
        train_loader, dev_loader, eval_loader = make_loaders(args)

        # Test loading one batch
        print("\n[Test] Loading one batch from train_loader...")
        batch = next(iter(train_loader))
        print(f"  - Waveforms shape: {batch['waveforms'].shape}")
        print(f"  - Labels shape: {batch['labels'].shape}")
        print(f"  - Labels: {batch['labels']}")
        print(f"  - First audio path: {batch['audio_paths'][0]}")

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {str(e)}")
        raise
