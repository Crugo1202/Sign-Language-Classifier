import torch
import torch.nn as nn
from typing import Dict, Tuple
from .tft_sign import TFTSignLite

class TFTSignLateFusion(TFTSignLite):
    def __init__(self, num_classes: int, cfg: Dict):
        super().__init__(num_classes, cfg)
        shared_dim = cfg["fusion"].get("shared_dim", 512)
        final_dim = cfg["fusion"].get("final_dim", 512)
        dropout = cfg["fusion"].get("dropout", 0.1)
        
        self.head_s = nn.Sequential(nn.Linear(shared_dim, final_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(final_dim, num_classes))
        self.head_t = nn.Sequential(nn.Linear(shared_dim, final_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(final_dim, num_classes))
        self.head_f = nn.Sequential(nn.Linear(shared_dim, final_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(final_dim, num_classes))
        self.head_d = nn.Sequential(nn.Linear(shared_dim, final_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(final_dim, num_classes))
        
        # 初始值設定為：手=1.0, 軌跡/身體=0.5, 臉=0.1, 動態=0.5
        self.modality_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.1, 0.5]))

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

        s_pool = self._masked_mean(s_seq, valid_mask)
        t_pool = self._masked_mean(t_seq, valid_mask)
        f_pool = self._masked_mean(f_seq, valid_mask)
        d_pool = self._masked_mean(d_seq, valid_mask)
        
        logits_s = self.head_s(s_pool)
        logits_t = self.head_t(t_pool)
        logits_f = self.head_f(f_pool)
        logits_d = self.head_d(d_pool)
        
        # 使用 ReLU 確保權重不會變成負的
        w = torch.relu(self.modality_weights)
        logits = (w[0] * logits_s) + (w[1] * logits_t) + (w[2] * logits_f) + (w[3] * logits_d)
        return logits, {}
