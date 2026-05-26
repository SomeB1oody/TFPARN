"""
AASIST model for ASVspoof5: the network architecture, config, and helpers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from typing import Union
import random


# ============================================================================
# AASIST Configuration
# ============================================================================

@dataclass
class AASISTArgs:
    """
    AASIST hyperparameters. Defaults match the original AASIST paper.
    """
    # First conv kernel size
    first_conv: int = 128

    # Filter sizes for the residual blocks
    filts: list = None  # filled in by __post_init__

    # Graph attention network dimensions
    gat_dims: list = None  # filled in by __post_init__

    # Graph pooling ratios
    pool_ratios: list = None  # filled in by __post_init__

    # Attention temperatures
    temperatures: list = None  # filled in by __post_init__

    # Number of output classes
    num_classes: int = 2  # binary: spoof (0) vs bonafide (1)

    def __post_init__(self):
        """Fill in the default lists if they weren't given"""
        if self.filts is None:
            # filts[0] is the Sinc conv output channels (an integer)
            # filts[1:] are [input, output] channels for each residual block
            self.filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        if self.gat_dims is None:
            self.gat_dims = [64, 32]

        if self.pool_ratios is None:
            self.pool_ratios = [0.5, 0.7, 0.5, 0.5]

        if self.temperatures is None:
            self.temperatures = [2.0, 2.0, 100.0, 100.0]

    def to_dict(self):
        """Return the config as a dict with the defaults filled in"""
        return {
            'first_conv': self.first_conv,
            'filts': self.filts,
            'gat_dims': self.gat_dims,
            'pool_ratios': self.pool_ratios,
            'temperatures': self.temperatures,
            'num_classes': self.num_classes
        }


# ============================================================================
# AASIST Model Components
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for a single (homogeneous) graph.
    Attention between nodes is built from pairwise products.
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # Attention map projection
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # Node projections
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        # Dropout for regularization
        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        # Temperature for the attention softmax
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features [batch_size, num_nodes, in_dim]

        Returns:
            Output features [batch_size, num_nodes, out_dim]
        """
        x = self.input_drop(x)

        # Build the attention map, then project the nodes with it
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)

        x = self._apply_BN(x)
        x = self.act(x)

        return x

    def _pairwise_mul_nodes(self, x):
        """
        Multiply every node with every other node, for the attention map.

        Args:
            x: [batch_size, num_nodes, dim]

        Returns:
            Pairwise products [batch_size, num_nodes, num_nodes, dim]
        """
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        """
        Build the attention map from the node features.

        Args:
            x: [batch_size, num_nodes, dim]

        Returns:
            Attention weights [batch_size, num_nodes, num_nodes, 1]
        """
        # Pairwise products, then project to attention scores
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)

        # Temperature scaling, then softmax over the source nodes
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        """
        Project the node features, both with and without attention.

        Args:
            x: Node features [batch_size, num_nodes, dim]
            att_map: Attention weights [batch_size, num_nodes, num_nodes, 1]

        Returns:
            Projected features [batch_size, num_nodes, out_dim]
        """
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        """Apply batch norm over the last dim"""
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        """Make a new parameter, Xavier-normal initialized"""
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    """
    Heterogeneous graph attention layer.
    Works on two node types, plus a master node, with separate attention per edge type.
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # Per-type projections
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # Attention projections
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        # One attention weight per edge type
        self.att_weight11 = self._init_new_params(out_dim, 1)  # type1 to type1
        self.att_weight22 = self._init_new_params(out_dim, 1)  # type2 to type2
        self.att_weight12 = self._init_new_params(out_dim, 1)  # type1 to type2 and back
        self.att_weightM = self._init_new_params(out_dim, 1)   # master node

        # Projections
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        # Temperature for the attention softmax
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        """
        Forward pass for the heterogeneous graph.

        Args:
            x1: Type 1 node features [batch_size, num_nodes1, dim]
            x2: Type 2 node features [batch_size, num_nodes2, dim]
            master: Master node features [batch_size, 1, dim] (optional)

        Returns:
            (x1_out, x2_out, master_out): updated features for each type
        """
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        # Per-type projections
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        # Stack both types together
        x = torch.cat([x1, x2], dim=1)

        # Start the master node as the mean if one wasn't passed in
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)

        # Build attention, update the master node, then project
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)

        x = self._apply_BN(x)
        x = self.act(x)

        # Split back into the two types
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):
        """Update the master node using its own attention"""
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        """Multiply every node with every other node, for attention"""
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        """Build the attention map for the master node"""
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        """
        Build the heterogeneous attention map, using a different weight
        for each edge type
        """
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        # Start with an empty attention board
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        # Fill each block with its own edge-type weight
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # Temperature scaling, then softmax
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        """Project the nodes, both with and without attention"""
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        """Project the master node, both with and without attention"""
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        """Apply batch norm over the last dim"""
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        """Make a new parameter, Xavier-normal initialized"""
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    """
    Graph pooling layer. Keeps the top-k nodes by a learned score.
    """

    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k  # pooling ratio
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        """
        Forward pass.

        Args:
            h: Node features [batch_size, num_nodes, dim]

        Returns:
            Pooled features [batch_size, num_nodes', dim], where num_nodes' is k * num_nodes
        """
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        """
        Keep the top-k nodes by score.

        Args:
            scores: Node scores [batch_size, num_nodes, 1]
            h: Node features [batch_size, num_nodes, dim]
            k: Fraction of nodes to keep

        Returns:
            The kept node features [batch_size, num_nodes', dim]
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)

        # Indices of the top-k scoring nodes
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        # Scale features by their score, then gather the top-k
        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class CONV(nn.Module):
    """
    Sinc convolution layer: learnable bandpass filters,
    a mel-scale filterbank applied in the time domain
    """

    @staticmethod
    def to_mel(hz):
        """Hz to mel"""
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        """Mel to Hz"""
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        mask=False
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (got {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Kernel size must be odd so the filter is symmetric
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # Set up the mel-scale filterbank
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        band_pass = torch.zeros(self.out_channels, self.kernel_size)

        # Build the bandpass filters
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        # Keep as a buffer so it moves with the model and isn't re-cloned each call
        self.register_buffer('band_pass', band_pass)

    def forward(self, x, mask=False):
        """
        Forward pass, with optional frequency masking for augmentation.

        Args:
            x: Input waveform [batch_size, 1, time]
            mask: Whether to randomly mask some frequencies (augmentation)

        Returns:
            Filtered output [batch_size, out_channels, time']
        """
        # Only clone the filters when masking; otherwise use the buffer as-is
        if mask:
            band_pass_filter = self.band_pass.clone()
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = self.band_pass

        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1
        )


class Residual_block(nn.Module):
    """
    Residual block of 2D convolutions.
    Used in the encoder to pull out spectro-temporal features.
    """

    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1
        )

        # Downsample the shortcut when the channel count changes
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        """Forward pass with the residual (skip) connection"""
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        # Downsample the identity to match if needed
        if self.downsample:
            identity = self.conv_downsample(identity)

        # Add the skip connection
        out += identity
        out = self.mp(out)

        return out


# ============================================================================
# AASIST Main Model
# ============================================================================

class AASIST(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks.

    The pipeline is:
    1. Sinc convolution for learnable filterbanks
    2. Residual encoder for spectro-temporal features
    3. Two graph attention branches, one spectral and one temporal
    4. Heterogeneous GAT to combine the two branches
    5. Graph pooling and a classification head
    """

    def __init__(self, args: AASISTArgs):
        super().__init__()

        self.args = args
        d_args = args.to_dict()

        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        # Sinc convolution (learnable mel-filterbank)
        self.conv_time = CONV(
            out_channels=filts[0],
            kernel_size=d_args["first_conv"],
            in_channels=1
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)

        # Dropout layers
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # Residual encoder: 6 blocks, sized from the filts config
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4]))
        )

        # Positional encoding for the spectral dimension
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))

        # Learnable master nodes
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Spectral GAT (GAT-S)
        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1],
            gat_dims[0],
            temperature=temperatures[0]
        )

        # Temporal GAT (GAT-T)
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1],
            gat_dims[0],
            temperature=temperatures[1]
        )

        # Heterogeneous GAT layers, two inference paths
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # Classification head
        self.out_layer = nn.Linear(5 * gat_dims[1], args.num_classes)

    def forward(self, x, Freq_aug=False):
        """
        Forward pass.

        Args:
            x: Input waveform [batch_size, channels, time]
            Freq_aug: Whether to apply frequency masking augmentation

        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Make sure the input is [B, C, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # Sinc convolution
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)  # [B, F, T] -> [B, 1, F, T]
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # Encoder pulls out spectro-temporal features
        e = self.encoder(x)  # [B, C, F, T]

        # === Spectral GAT (GAT-S) ===
        # Max-pool over time
        e_S, _ = torch.max(torch.abs(e), dim=3)  # [B, C, F]
        e_S = e_S.transpose(1, 2)  # [B, F, C]
        e_S = e_S + self.pos_S  # add positional encoding

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # [B, F', C']

        # === Temporal GAT (GAT-T) ===
        # Max-pool over frequency
        e_T, _ = torch.max(torch.abs(e), dim=2)  # [B, C, T]
        e_T = e_T.transpose(1, 2)  # [B, T, C]

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)  # [B, T', C']

        # Learnable master nodes
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # === Inference Path 1 ===
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        # Skip connection
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # === Inference Path 2 ===
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        # Skip connection
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # Dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        # Take the element-wise max across the two paths
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Pool each branch down to a single vector
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        # Concatenate everything
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        # Classify
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


# ============================================================================
# Model Creation
# ============================================================================

def create_aasist_model(args: AASISTArgs = None, device: torch.device = None) -> AASIST:
    """
    Build an AASIST model from the given config.

    Args:
        args: AASIST configuration (uses defaults if None)
        device: Device to put the model on

    Returns:
        AASIST model instance
    """
    if args is None:
        args = AASISTArgs()

    print(f"\n[Model] Creating AASIST model")
    print(f"  - Configuration:")
    print(f"    - First conv kernel: {args.first_conv}")
    print(f"    - Filter dimensions: {args.filts}")
    print(f"    - GAT dimensions: {args.gat_dims}")
    print(f"    - Pool ratios: {args.pool_ratios}")
    print(f"    - Temperatures: {args.temperatures}")
    print(f"    - Number of classes: {args.num_classes}")

    model = AASIST(args)

    if device is not None:
        model = model.to(device)
        print(f"  - Model moved to device: {device}")

    # Count the trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")

    return model
