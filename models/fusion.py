from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandShapeFusion(nn.Module):
    """
    Cross-attention fusion between right and left handshape streams.
    """

    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_r2l = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.cross_l2r = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(
        self,
        right_seq: torch.Tensor,
        left_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # right_seq, left_seq: (B, T, 512)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        r2l, _ = self.cross_r2l(right_seq, left_seq, left_seq, key_padding_mask=key_padding_mask)
        l2r, _ = self.cross_l2r(left_seq, right_seq, right_seq, key_padding_mask=key_padding_mask)
        r2l = self.dropout(r2l)
        l2r = self.dropout(l2r)
        if valid_mask is None:
            pooled_r = r2l.mean(dim=1)
            pooled_l = l2r.mean(dim=1)
        else:
            w = valid_mask.to(r2l.dtype).unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1.0)
            pooled_r = (r2l * w).sum(dim=1) / denom
            pooled_l = (l2r * w).sum(dim=1) / denom
        fused = torch.cat([pooled_r, pooled_l], dim=-1)
        return self.proj(self.dropout(fused))


class ThreeWayFusion(nn.Module):
    """
    Three-way cross-modal attention and feature projection.
    """

    def __init__(
        self,
        s_dim: int = 512,
        t_dim: int = 512,
        f_dim: int = 256,
        shared_dim: int = 512,
        modal_heads: int = 8,
        final_dim: int = 512,
        dropout: float = 0.0,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        self.modality_dropout_p = modality_dropout
        self.dropout = nn.Dropout(dropout)
        self.proj_s = nn.Linear(s_dim, shared_dim)
        self.proj_t = nn.Linear(t_dim, shared_dim)
        self.proj_f = nn.Linear(f_dim, shared_dim)
        self.norm_s = nn.LayerNorm(shared_dim)
        self.norm_t = nn.LayerNorm(shared_dim)
        self.norm_f = nn.LayerNorm(shared_dim)

        self.attn_s = nn.MultiheadAttention(
            shared_dim, modal_heads, dropout=dropout, batch_first=True
        )
        self.attn_t = nn.MultiheadAttention(
            shared_dim, modal_heads, dropout=dropout, batch_first=True
        )
        self.attn_f = nn.MultiheadAttention(
            shared_dim, modal_heads, dropout=dropout, batch_first=True
        )

        self.final_proj = nn.Linear(shared_dim * 3, final_dim)

    def _modality_dropout(
        self, s: torch.Tensor, t: torch.Tensor, f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_p <= 0.0:
            return s, t, f
        keep_prob = 1.0 - self.modality_dropout_p
        mask = torch.empty(s.size(0), 3, 1, device=s.device).bernoulli_(keep_prob)
        empty = mask.sum(dim=1, keepdim=True) == 0
        if empty.any():
            empty_rows = empty.squeeze(-1).squeeze(-1)
            rows = torch.arange(s.size(0), device=s.device)[empty_rows]
            fallback = torch.randint(0, 3, (rows.numel(),), device=s.device)
            mask[empty_rows] = 0.0
            mask[rows, fallback, 0] = 1.0
        stacked = torch.stack([s, t, f], dim=1) * mask / keep_prob
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]

    def _enhance(
        self,
        source: torch.Tensor,
        context_a: torch.Tensor,
        context_b: torch.Tensor,
        layer: nn.MultiheadAttention,
    ) -> torch.Tensor:
        # Each modality is a global vector; treat as len-1 token for attention.
        q = source.unsqueeze(1)
        ctx = torch.stack([context_a, context_b], dim=1)
        out, _ = layer(q, ctx, ctx)
        return source + self.dropout(out.squeeze(1))

    def forward(
        self, f_s: torch.Tensor, f_t: torch.Tensor, f_f: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = self.dropout(self.norm_s(self.proj_s(f_s)))
        t = self.dropout(self.norm_t(self.proj_t(f_t)))
        f = self.dropout(self.norm_f(self.proj_f(f_f)))
        s, t, f = self._modality_dropout(s, t, f)

        s_enh = self._enhance(s, t, f, self.attn_s)
        t_enh = self._enhance(t, s, f, self.attn_t)
        f_enh = self._enhance(f, s, t, self.attn_f)

        final = self.final_proj(self.dropout(torch.cat([s_enh, t_enh, f_enh], dim=-1)))
        return final, {"s_enhanced": s_enh, "t_enhanced": t_enh, "f_enhanced": f_enh}


class GeometricConsistency(nn.Module):
    def __init__(self, dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.head_a = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj_dim))
        self.head_b = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj_dim))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        p1 = self.head_a(x1)
        p2 = self.head_b(x2)
        return 1.0 - F.cosine_similarity(p1, p2, dim=-1).mean()
