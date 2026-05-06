import torch
import torch.nn as nn
from typing import Dict, Tuple
from .tft_sign import TFTSignLite

class TFTSignHandOnly(TFTSignLite):
    def __init__(self, num_classes: int, cfg: Dict):
        super().__init__(num_classes, cfg)
        shared_dim = cfg["fusion"].get("shared_dim", 512)
        self.head = nn.Sequential(
            nn.Linear(shared_dim, cfg["fusion"].get("final_dim", 512)),
            nn.GELU(),
            nn.Dropout(cfg["fusion"].get("dropout", 0.1)),
            nn.Linear(cfg["fusion"].get("final_dim", 512), num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        right_shape = batch["right_hand_shape"]
        left_shape = batch["left_hand_shape"]
        valid_mask = batch.get("valid_mask", None)

        right_seq, right_pool = self.right_tssn(right_shape, valid_mask=valid_mask)
        left_seq, left_pool = self.left_tssn(left_shape, valid_mask=valid_mask)
        s_seq = self.input_dropout(self.norm_s_in(self.proj_s(torch.cat([right_seq, left_seq], dim=-1))))

        pooled = self._masked_mean(s_seq, valid_mask)
        logits = self.head(pooled)
        return logits, {}
