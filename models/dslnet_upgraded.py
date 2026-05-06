from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ftde import FTDEDual
from .fusion import HandShapeFusion, ThreeWayFusion
from .nmn import NonManualNetwork
from .tssn import TSSNEncoder


class DSLNetUpgraded(nn.Module):
    """
    Upgraded DSLNet:
    - Dual-hand TSSN (right + left)
    - FTDE-Dual trajectory encoder
    - NMN face encoder
    - 3-way fusion
    """

    def __init__(self, num_classes: int, cfg: Dict):
        super().__init__()
        tssn_cfg = cfg["tssn"]
        ftde_cfg = cfg["ftde"]
        nmn_cfg = cfg["nmn"]
        fusion_cfg = cfg["fusion"]
        dropout = fusion_cfg.get("dropout", 0.0)
        modality_dropout = fusion_cfg.get("modality_dropout", 0.0)

        self.right_tssn = TSSNEncoder(
            k=tssn_cfg["k"],
            t_kernel=tssn_cfg["temporal_kernel"],
            hidden_size=tssn_cfg["hidden_size"],
            lstm_layers=tssn_cfg["lstm_layers"],
            attn_heads=tssn_cfg["attn_heads"],
            output_dim=tssn_cfg["output_dim"],
            dropout=dropout,
        )
        self.left_tssn = TSSNEncoder(
            k=tssn_cfg["k"],
            t_kernel=tssn_cfg["temporal_kernel"],
            hidden_size=tssn_cfg["hidden_size"],
            lstm_layers=tssn_cfg["lstm_layers"],
            attn_heads=tssn_cfg["attn_heads"],
            output_dim=tssn_cfg["output_dim"],
            dropout=dropout,
        )

        self.hand_fusion = HandShapeFusion(
            dim=tssn_cfg["output_dim"], heads=fusion_cfg["hand_heads"], dropout=dropout
        )
        self.ftde = FTDEDual(
            conv_channels=ftde_cfg["conv_channels"],
            conv_kernel=ftde_cfg["conv_kernel"],
            hidden_size=ftde_cfg["hidden_size"],
            lstm_layers=ftde_cfg["lstm_layers"],
            alpha_finsler=ftde_cfg["alpha_finsler"],
            temperature_tau=ftde_cfg["temperature_tau"],
            dropout=dropout,
        )
        self.nmn = NonManualNetwork(
            grouped_channels=nmn_cfg["grouped_channels"],
            temporal_channels=nmn_cfg["temporal_channels"],
            temporal_kernel=nmn_cfg["temporal_kernel"],
            hidden_size=nmn_cfg["hidden_size"],
            dropout=dropout,
        )
        s_dim = tssn_cfg["output_dim"]
        t_dim = ftde_cfg["hidden_size"] * 2
        f_dim = nmn_cfg["hidden_size"] * 2
        self.shared_dim = fusion_cfg["shared_dim"]
        self.fusion = ThreeWayFusion(
            s_dim=s_dim,
            t_dim=t_dim,
            f_dim=f_dim,
            shared_dim=fusion_cfg["shared_dim"],
            modal_heads=fusion_cfg["modal_heads"],
            final_dim=fusion_cfg["final_dim"],
            dropout=dropout,
            modality_dropout=modality_dropout,
        )
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_cfg["final_dim"], num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        right_shape = batch["right_hand_shape"]
        left_shape = batch["left_hand_shape"]
        dual_traj = batch["dual_wrist_traj"]
        pose = batch["pose_landmarks"]
        face = batch["face_landmarks_norm"]
        valid_mask = batch.get("valid_mask", None)

        right_seq, right_pool = self.right_tssn(right_shape, valid_mask=valid_mask)
        left_seq, left_pool = self.left_tssn(left_shape, valid_mask=valid_mask)
        f_s = self.hand_fusion(right_seq, left_seq, valid_mask=valid_mask)

        f_t, _ = self.ftde(pose, valid_mask=valid_mask)
        f_f, _ = self.nmn(face, valid_mask=valid_mask)

        f_final, enhanced = self.fusion(f_s, f_t, f_f)  # (B, 512)
        logits = self.classifier(self.classifier_dropout(f_final))

        enhanced["right_hand_pooled"] = right_pool
        enhanced["left_hand_pooled"] = left_pool
        return logits, enhanced
