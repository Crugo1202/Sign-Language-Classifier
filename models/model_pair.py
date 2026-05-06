import torch
import torch.nn as nn
from typing import Dict, Tuple
from .tft_sign import TFTSignLite

class PairwiseCrossModalBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_sf = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_st = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_sf = nn.LayerNorm(dim)
        self.norm_st = nn.LayerNorm(dim)

    def forward(self, s, t, f, d, valid_mask):
        key_padding_mask = None
        if valid_mask is not None:
             key_padding_mask = ~valid_mask.bool()
        
        out_sf, _ = self.attn_sf(s, f, f, key_padding_mask=key_padding_mask)
        s_f = self.norm_sf(s + out_sf)
        
        out_st, _ = self.attn_st(s, t, t, key_padding_mask=key_padding_mask)
        s_t = self.norm_st(s + out_st)
        
        return torch.cat([s, s_f, s_t], dim=-1)

class TFTSignPairwise(TFTSignLite):
    def __init__(self, num_classes: int, cfg: Dict):
        super().__init__(num_classes, cfg)
        shared_dim = cfg["fusion"].get("shared_dim", 512)
        modal_heads = cfg["fusion"].get("modal_heads", 8)
        dropout = cfg["fusion"].get("dropout", 0.1)
        self.cross_block = PairwiseCrossModalBlock(shared_dim, heads=modal_heads, dropout=dropout)
        
        self.head = nn.Sequential(
            nn.Linear(shared_dim * 3, cfg["fusion"].get("final_dim", 512)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg["fusion"].get("final_dim", 512), num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        right_shape = batch["right_hand_shape"]
        left_shape = batch["left_hand_shape"]
        dual_traj = batch["dual_wrist_traj"]
        pose = batch["pose_landmarks"]
        face = batch["face_landmarks_norm"]
        hand_orientation = batch.get("hand_orientation", None)
        valid_mask = batch.get("valid_mask", None)

        right_seq, right_pool = self.right_tssn(right_shape, valid_mask=valid_mask)
        left_seq, left_pool = self.left_tssn(left_shape, valid_mask=valid_mask)
        s_seq = self.input_dropout(self.norm_s_in(self.proj_s(torch.cat([right_seq, left_seq], dim=-1))))

        _, t_seq = self.ftde(pose, valid_mask=valid_mask)
        t_seq = self.input_dropout(self.norm_t_in(self.proj_t(t_seq)))

        _, f_seq = self.nmn(face, valid_mask=valid_mask)
        f_seq = self.input_dropout(self.norm_f_in(self.proj_f(f_seq)))

        d_seq = self.dyn_mlp(self._dynamics(dual_traj))
        if hand_orientation is not None:
            d_seq = d_seq + self.orientation_mlp(hand_orientation)
        d_seq = self.input_dropout(self.norm_d_in(d_seq))

        features = self.cross_block(s_seq, t_seq, f_seq, d_seq, valid_mask)
        pooled = self._masked_mean(features, valid_mask)
        logits = self.head(pooled)
        return logits, {}
