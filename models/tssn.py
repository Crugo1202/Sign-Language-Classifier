from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(sequence: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return sequence.mean(dim=1)
    weights = valid_mask.to(sequence.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


def build_knn_graph(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    x: (N, V, C)
    returns adjacency: (N, V, V)
    """
    n, v, _ = x.shape
    dist = torch.cdist(x, x)
    _, idx = dist.topk(k=min(k, v), dim=-1, largest=False)
    adj = torch.zeros((n, v, v), device=x.device, dtype=x.dtype)
    adj.scatter_(-1, idx, 1.0)
    adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
    return adj


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 8):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, V, C)
        adj = build_knn_graph(x, self.k)
        msg = torch.matmul(adj, x)
        out = self.proj(msg)
        return self.norm(out)


class STGCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 8, t_kernel: int = 9):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, k=k)
        self.temporal = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=t_kernel,
            stride=1,
            padding=t_kernel // 2,
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C)
        b, t, v, c = x.shape
        x_flat = x.reshape(b * t, v, c)
        s = self.spatial(x_flat).reshape(b, t, v, -1)

        s_perm = s.permute(0, 2, 3, 1).reshape(b * v, s.shape[-1], t)
        y = self.temporal(s_perm)
        y = self.bn(y)
        y = y.reshape(b, v, s.shape[-1], t).permute(0, 3, 1, 2)

        r = self.residual(x)
        return self.act(y + r)


class TSSNEncoder(nn.Module):
    """
    Topology-aware Spatiotemporal Network for handshape stream.
    Input: (B, T, 21, 3)
    Outputs:
      seq_feat: (B, T, output_dim)
      pooled_feat: (B, output_dim)
    """

    def __init__(
        self,
        k: int = 8,
        t_kernel: int = 9,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        attn_heads: int = 8,
        output_dim: int = 512,
    ):
        super().__init__()
        channels = [3, 64, 128, 256, 256]
        self.blocks = nn.ModuleList(
            [
                STGCBlock(channels[0], channels[1], k=k, t_kernel=t_kernel),
                STGCBlock(channels[1], channels[2], k=k, t_kernel=t_kernel),
                STGCBlock(channels[2], channels[3], k=k, t_kernel=t_kernel),
                STGCBlock(channels[3], channels[4], k=k, t_kernel=t_kernel),
            ]
        )
        self.multi_proj = nn.Linear(128 + 256 + 256, 256)
        self.temporal_model = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=attn_heads, batch_first=True
        )
        self.out_proj = nn.Linear(hidden_size * 2, output_dim)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 21, 3)
        feats: List[torch.Tensor] = []
        y = x
        for i, block in enumerate(self.blocks):
            y = block(y)
            if i in (1, 2, 3):
                feats.append(y.mean(dim=2))

        ms = torch.cat(feats, dim=-1)  # (B, T, 640)
        ms = self.multi_proj(ms)  # (B, T, 256)
        lstm_out, _ = self.temporal_model(ms)  # (B, T, 512)

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        attn_out, _ = self.temporal_attn(
            lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask
        )
        seq_feat = self.out_proj(attn_out)  # (B, T, output_dim)
        pooled = masked_mean(seq_feat, valid_mask)
        return seq_feat, pooled
