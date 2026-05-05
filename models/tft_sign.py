from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ftde import FTDEDual
from .nmn import NonManualNetwork
from .tssn import TSSNEncoder


class CrossModalBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_s = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_f = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_d = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)
        self.norm_d = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def _update(
        self,
        q_seq: torch.Tensor,
        ctx_a: torch.Tensor,
        ctx_b: torch.Tensor,
        ctx_c: torch.Tensor,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        ctx = torch.cat([ctx_a, ctx_b, ctx_c], dim=1)
        out, _ = attn(q_seq, ctx, ctx)
        z = norm(q_seq + out)
        return self.ffn_norm(z + self.ffn(z))

    def forward(
        self,
        s_seq: torch.Tensor,
        t_seq: torch.Tensor,
        f_seq: torch.Tensor,
        d_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self._update(s_seq, t_seq, f_seq, d_seq, self.attn_s, self.norm_s)
        t = self._update(t_seq, s_seq, f_seq, d_seq, self.attn_t, self.norm_t)
        f = self._update(f_seq, s_seq, t_seq, d_seq, self.attn_f, self.norm_f)
        d = self._update(d_seq, s_seq, t_seq, f_seq, self.attn_d, self.norm_d)
        return s, t, f, d


class TFTSignLite(nn.Module):
    @staticmethod
    def _masked_mean(seq: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            return seq.mean(dim=1)
        weights = valid_mask.to(seq.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (seq * weights).sum(dim=1) / denom

    """
    A compact multimodal model with:
    - Hand shape stream (dual TSSN)
    - Wrist trajectory stream (FTDE)
    - Face stream (NMN)
    - Derived dynamics stream (velocity + acceleration)
    - Time-aware cross-modal attention fusion
    """

    def __init__(self, num_classes: int, cfg: Dict):
        super().__init__()
        tssn_cfg = cfg["tssn"]
        ftde_cfg = cfg["ftde"]
        nmn_cfg = cfg["nmn"]
        fusion_cfg = cfg["fusion"]
        dyn_cfg = cfg.get("dynamic", {})
        orient_cfg = cfg.get("orientation", {})
        dropout = fusion_cfg.get("dropout", 0.1)
        self.modality_dropout_p = fusion_cfg.get("modality_dropout", 0.0)

        self.right_tssn = TSSNEncoder(
            k=tssn_cfg["k"],
            t_kernel=tssn_cfg["temporal_kernel"],
            hidden_size=tssn_cfg["hidden_size"],
            lstm_layers=tssn_cfg["lstm_layers"],
            attn_heads=tssn_cfg["attn_heads"],
            output_dim=tssn_cfg["output_dim"],
        )
        self.left_tssn = TSSNEncoder(
            k=tssn_cfg["k"],
            t_kernel=tssn_cfg["temporal_kernel"],
            hidden_size=tssn_cfg["hidden_size"],
            lstm_layers=tssn_cfg["lstm_layers"],
            attn_heads=tssn_cfg["attn_heads"],
            output_dim=tssn_cfg["output_dim"],
        )
        self.ftde = FTDEDual(
            conv_channels=ftde_cfg["conv_channels"],
            conv_kernel=ftde_cfg["conv_kernel"],
            hidden_size=ftde_cfg["hidden_size"],
            lstm_layers=ftde_cfg["lstm_layers"],
            alpha_finsler=ftde_cfg["alpha_finsler"],
            temperature_tau=ftde_cfg["temperature_tau"],
        )
        self.nmn = NonManualNetwork(
            grouped_channels=nmn_cfg["grouped_channels"],
            temporal_channels=nmn_cfg["temporal_channels"],
            temporal_kernel=nmn_cfg["temporal_kernel"],
            hidden_size=nmn_cfg["hidden_size"],
        )

        shared_dim = fusion_cfg.get("shared_dim", 512)
        modal_heads = fusion_cfg.get("modal_heads", 8)
        self.proj_s = nn.Linear(tssn_cfg["output_dim"] * 2, shared_dim)
        self.proj_t = nn.Linear(ftde_cfg["hidden_size"] * 2, shared_dim)
        self.proj_f = nn.Linear(nmn_cfg["hidden_size"] * 2, shared_dim)
        self.norm_s_in = nn.LayerNorm(shared_dim)
        self.norm_t_in = nn.LayerNorm(shared_dim)
        self.norm_f_in = nn.LayerNorm(shared_dim)
        self.norm_d_in = nn.LayerNorm(shared_dim)
        self.input_dropout = nn.Dropout(dropout)

        dyn_hidden = dyn_cfg.get("hidden_dim", 128)
        self.dyn_mlp = nn.Sequential(
            nn.Linear(18, dyn_hidden),
            nn.GELU(),
            nn.Linear(dyn_hidden, shared_dim),
        )
        orient_hidden = orient_cfg.get("hidden_dim", 64)
        self.orientation_mlp = nn.Sequential(
            nn.Linear(8, orient_hidden),
            nn.GELU(),
            nn.Linear(orient_hidden, shared_dim),
        )

        self.cross_block = CrossModalBlock(shared_dim, heads=modal_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(shared_dim * 4, fusion_cfg.get("final_dim", 512)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_cfg.get("final_dim", 512), num_classes),
        )

    @staticmethod
    def _dynamics(dual_traj: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(dual_traj)
        vel[:, 1:] = dual_traj[:, 1:] - dual_traj[:, :-1]
        acc = torch.zeros_like(vel)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        return torch.cat([dual_traj, vel, acc], dim=-1)

    def _modality_dropout(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout_p <= 0.0:
            return seq
        keep_prob = 1.0 - self.modality_dropout_p
        mask = torch.empty(seq.size(0), 1, 1, device=seq.device).bernoulli_(keep_prob)
        return seq * mask / keep_prob

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        right_shape = batch["right_hand_shape"]
        left_shape = batch["left_hand_shape"]
        dual_traj = batch["dual_wrist_traj"]
        face = batch["face_landmarks_norm"]
        hand_orientation = batch.get("hand_orientation", None)
        valid_mask = batch.get("valid_mask", None)

        right_seq, right_pool = self.right_tssn(right_shape, valid_mask=valid_mask)
        left_seq, left_pool = self.left_tssn(left_shape, valid_mask=valid_mask)
        s_seq = self.input_dropout(self.norm_s_in(self.proj_s(torch.cat([right_seq, left_seq], dim=-1))))

        _, t_seq = self.ftde(dual_traj, valid_mask=valid_mask)
        t_seq = self.input_dropout(self.norm_t_in(self.proj_t(t_seq)))

        _, f_seq = self.nmn(face, valid_mask=valid_mask)
        f_seq = self.input_dropout(self.norm_f_in(self.proj_f(f_seq)))

        d_seq = self.dyn_mlp(self._dynamics(dual_traj))
        if hand_orientation is not None:
            d_seq = d_seq + self.orientation_mlp(hand_orientation)
        d_seq = self.input_dropout(self.norm_d_in(d_seq))

        s_seq = self._modality_dropout(s_seq)
        t_seq = self._modality_dropout(t_seq)
        f_seq = self._modality_dropout(f_seq)
        d_seq = self._modality_dropout(d_seq)
        s_enh, t_enh, f_enh, d_enh = self.cross_block(s_seq, t_seq, f_seq, d_seq)

        pooled = torch.cat(
            [
                self._masked_mean(s_enh, valid_mask),
                self._masked_mean(t_enh, valid_mask),
                self._masked_mean(f_enh, valid_mask),
                self._masked_mean(d_enh, valid_mask),
            ],
            dim=-1,
        )
        logits = self.head(pooled)
        aux = {
            "s_enhanced": self._masked_mean(s_enh, valid_mask),
            "t_enhanced": self._masked_mean(t_enh, valid_mask),
            "f_enhanced": self._masked_mean(f_enh, valid_mask),
            "right_hand_pooled": right_pool,
            "left_hand_pooled": left_pool,
        }
        return logits, aux
