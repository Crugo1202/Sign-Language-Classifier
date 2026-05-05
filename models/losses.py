from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .fusion import GeometricConsistency


class DSLNetUpgradedLoss(nn.Module):
    def __init__(
        self,
        alpha_geo: float = 0.1,
        beta_sym: float = 0.05,
        label_smoothing: float = 0.1,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.alpha_geo = alpha_geo
        self.beta_sym = beta_sym
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.geo_st = GeometricConsistency(dim=feature_dim, proj_dim=128)
        self.geo_sf = GeometricConsistency(dim=feature_dim, proj_dim=128)
        self.geo_tf = GeometricConsistency(dim=feature_dim, proj_dim=128)
        self.sym = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        l_ce = self.ce(logits, labels)

        s_enh = aux["s_enhanced"]
        t_enh = aux["t_enhanced"]
        f_enh = aux["f_enhanced"]

        l_geo_st = self.geo_st(s_enh, t_enh)
        l_geo_sf = self.geo_sf(s_enh, f_enh)
        l_geo_tf = self.geo_tf(t_enh, f_enh)
        l_geo = l_geo_st + l_geo_sf + l_geo_tf

        l_sym = self.sym(aux["right_hand_pooled"], aux["left_hand_pooled"])
        total = l_ce + self.alpha_geo * l_geo + self.beta_sym * l_sym

        return {
            "total": total,
            "ce": l_ce,
            "geo": l_geo,
            "sym": l_sym,
            "geo_st": l_geo_st,
            "geo_sf": l_geo_sf,
            "geo_tf": l_geo_tf,
        }
