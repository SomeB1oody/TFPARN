"""
Transformer model for binary speech classification on ASVspoof5.
Pipeline: mel spectrogram, then Transformer encoder, then a binary classifier.

Labels: spoof (AI-generated) is 0, bonafide (real human) is 1.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn


# Model config
@dataclass
class SpeechClassifierArgs:
    """
    Settings for the speech Transformer classifier
    """
    # Mel spectrogram settings
    n_mels: int = 128
    n_fft: int = 768
    hop_length: int = 160
    sample_rate: int = 16000

    # Transformer settings
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.3
    activation: str = "relu"  # "relu" or "gelu"

    # Pooling: "mean", "attention", or "top-k"
    pooling_method: str = "mean"
    top_k_ratio: float = 0.3  # for top-k pooling, fraction of frames to keep


# Model components
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the Transformer.

    Args:
        d_model: model dimension
        max_len: maximum sequence length
        dropout: dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Store as a buffer so it isn't trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Main model
class SpeechTransformerClassifier(nn.Module):
    """
    Transformer-based binary speech classifier.

    Steps:
        1. Frontend: log-mel spectrogram, then a linear projection
        2. Positional encoding
        3. Transformer encoder (multi-layer, multi-head attention)
        4. Pooling: mean, attention, or top-k, with masking
        5. Classification head: linear layers giving [B, 2] logits

    Input:
        - waveforms: FloatTensor [B, C, T] (raw waveform)
        - lengths: LongTensor [B] (actual lengths)

    Output:
        - logits: FloatTensor [B, 2]
          - logits[:, 0]: spoof (AI-generated)
          - logits[:, 1]: bonafide (real human)
    """

    def __init__(self, args: SpeechClassifierArgs):
        super().__init__()

        self.n_mels = args.n_mels
        self.n_fft = args.n_fft
        self.hop_length = args.hop_length
        self.sample_rate = args.sample_rate
        self.d_model = args.d_model
        self.pooling_method = args.pooling_method
        self.top_k_ratio = args.top_k_ratio

        print(f"\n[Model] Initializing SpeechTransformerClassifier")
        print(f"  - n_mels: {args.n_mels}")
        print(f"  - n_fft: {args.n_fft}")
        print(f"  - hop_length: {args.hop_length}")
        print(f"  - d_model: {args.d_model}")
        print(f"  - nhead: {args.nhead}")
        print(f"  - num_layers: {args.num_layers}")
        print(f"  - dim_feedforward: {args.dim_feedforward}")
        print(f"  - dropout: {args.dropout}")
        print(f"  - activation: {args.activation}")
        print(f"  - pooling_method: {args.pooling_method}")

        # Build the mel filterbank and keep it as a buffer
        mel_basis = self._create_mel_filterbank(
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            sample_rate=args.sample_rate,
            fmin=0.0,
            fmax=args.sample_rate // 2
        )
        self.register_buffer('mel_basis', mel_basis)

        # Keep the Hann window as a buffer too
        self.register_buffer('hann', torch.hann_window(args.n_fft))

        # Normalize the input
        self.input_norm = nn.LayerNorm(args.n_mels)

        # Project features from [B, T', n_mels] to [B, T', d_model]
        self.feature_projection = nn.Linear(args.n_mels, args.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            args.d_model,
            max_len=5000,
            dropout=args.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_layers
        )

        # Attention pooling layer, only when used
        if self.pooling_method == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(args.d_model, args.d_model // 2),
                nn.Tanh(),
                nn.Linear(args.d_model // 2, 1)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, 2)  # 2 classes: spoof is 0, bonafide is 1
        )

        self._reset_parameters()
        print(f"[✓] Model initialized successfully")

    def _reset_parameters(self):
        """Initialize the parameters"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'feature_projection' in name:
                    nn.init.xavier_uniform_(p, gain=0.5)
                else:
                    nn.init.xavier_uniform_(p)

    def _create_mel_filterbank(
        self,
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        fmin: float,
        fmax: float
    ) -> torch.Tensor:
        """
        Build the mel filterbank.

        Returns:
            [n_mels, n_fft // 2 + 1]
        """

        def hz_to_mel(hz):
            return 2595 * torch.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Pick evenly spaced points on the mel scale
        mel_min = hz_to_mel(torch.tensor(fmin))
        mel_max = hz_to_mel(torch.tensor(fmax))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Map them to FFT bins
        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

        # Build the triangular filters
        fbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(1, n_mels + 1):
            f_left = bin_points[m - 1].item()
            f_center = bin_points[m].item()
            f_right = bin_points[m + 1].item()

            # Going up
            for k in range(f_left, f_center):
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)

            # Coming down
            for k in range(f_center, f_right):
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return fbank

    def _compute_mel_spectrogram(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-mel spectrogram.

        Args:
            waveforms: [B, 1, T] mono waveform

        Returns:
            [B, T', n_mels] where T' is roughly T // hop_length
        """
        # Drop the channel dim: [B, 1, T] becomes [B, T]
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        # STFT, reusing the precomputed Hann window buffer
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.hann,
            center=True,
            return_complex=True
        )  # [B, n_fft // 2 + 1, T']

        # Power spectrogram
        power_spec = torch.abs(stft) ** 2  # [B, n_fft // 2 + 1, T']

        # Apply the mel filterbank
        mel_spec = torch.matmul(
            self.mel_basis.to(power_spec.device),
            power_spec
        )  # [B, n_mels, T']

        # Move to a log (dB) scale
        mel_spec = torch.clamp(mel_spec, min=1e-10)
        log_mel_spec = 10.0 * torch.log10(mel_spec + 1e-10)
        log_mel_spec = torch.clamp(log_mel_spec, min=-80.0, max=0.0) / 10.0

        # Transpose to [B, T', n_mels]
        log_mel_spec = log_mel_spec.transpose(1, 2)

        return log_mel_spec

    def _create_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """
        Build a padding mask.

        Args:
            lengths: [B] actual lengths
            max_len: maximum length

        Returns:
            [B, max_len] bool tensor where True marks padding positions
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            waveforms: FloatTensor [B, 1, T] preprocessed mono waveform

        Returns:
            logits: FloatTensor [B, 2]
                   - logits[:, 0]: spoof score
                   - logits[:, 1]: bonafide score
        """
        # 1. Get the log-mel spectrogram
        log_mel = self._compute_mel_spectrogram(waveforms)  # [B, T', n_mels]

        # 2. Normalize the input
        log_mel = self.input_norm(log_mel)  # [B, T', n_mels]

        # 3. Project the features
        features = self.feature_projection(log_mel)  # [B, T', d_model]

        # 4. Add positional encoding
        features = self.pos_encoder(features)  # [B, T', d_model]

        # 5. Build the padding mask
        # With fixed-length crops and center=True in the STFT, the whole
        # feature length is treated as valid
        feature_lengths = torch.full((features.size(0),), features.size(1),
                                     device=features.device, dtype=torch.long)
        src_key_padding_mask = self._create_padding_mask(
            feature_lengths,
            features.size(1)
        )  # [B, T']

        # 6. Transformer encoder
        encoded = self.transformer_encoder(
            features,
            src_key_padding_mask=src_key_padding_mask
        )  # [B, T', d_model]

        # 7. Pool over time, respecting the mask
        mask_expanded = (~src_key_padding_mask).unsqueeze(-1)  # [B, T', 1]

        if self.pooling_method == "mean":
            # Masked mean pooling
            masked_encoded = encoded * mask_expanded  # [B, T', d_model]
            pooled = masked_encoded.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # [B, d_model]

        elif self.pooling_method == "attention":
            # Attention pooling
            attention_scores = self.attention_pooling(encoded)  # [B, T', 1]
            attention_scores = attention_scores.masked_fill(src_key_padding_mask.unsqueeze(-1), float('-inf'))
            attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T', 1]
            pooled = (encoded * attention_weights).sum(dim=1)  # [B, d_model]

        elif self.pooling_method == "top-k":
            # Top-k pooling: keep the frames with the largest L2 norm
            frame_norms = torch.norm(encoded, p=2, dim=-1)  # [B, T']
            frame_norms = frame_norms.masked_fill(src_key_padding_mask, float('-inf'))

            # Work out k from the actual lengths
            valid_lengths = mask_expanded.sum(dim=1).squeeze(-1)  # [B]
            k_values = (valid_lengths * self.top_k_ratio).clamp(min=1).long()  # [B]

            # Top-k indices per sample
            batch_size = encoded.size(0)
            pooled = torch.zeros(batch_size, self.d_model, device=encoded.device)

            for i in range(batch_size):
                k = k_values[i].item()
                valid_len = valid_lengths[i].long().item()

                # Top-k frames for this sample
                _, topk_indices = torch.topk(frame_norms[i, :valid_len], k=min(k, valid_len))

                # Average them
                pooled[i] = encoded[i, topk_indices].mean(dim=0)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # 8. Classify
        logits = self.classifier(pooled)  # [B, 2]

        return logits


# Build the model
def create_model(args: SpeechClassifierArgs = None) -> SpeechTransformerClassifier:
    """
    Create a model instance.

    Args:
        args: model config (uses the default if None)

    Returns:
        a SpeechTransformerClassifier
    """
    if args is None:
        args = SpeechClassifierArgs()

    return SpeechTransformerClassifier(args)
