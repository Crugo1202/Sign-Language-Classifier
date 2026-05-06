from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonManualNetwork(nn.Module):
    """
    Face-based non-manual feature encoder.
    Input: (B, T, K, 3), K is semantic facial keypoints (e.g. 60-80)
    Output:
      pooled feature: (B, 256)
      sequence feature: (B, T, 256)
    """

    def __init__(
        self,
        num_keypoints: int = 478,  # MediaPipe usually outputs 478 face keypoints
        grouped_channels: int = 16,
        temporal_channels: int = 128,
        temporal_kernel: int = 5,
        hidden_size: int = 128,
        dropout: float = 0.0,
        attn_dropout: float | None = None,
    ):
        super().__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        
        # 1. Spatial: 這裡保留你的 1x1 Conv
        self.spatial = nn.Conv2d(
            in_channels=6, out_channels=grouped_channels, kernel_size=1, stride=1, bias=False
        )
        self.spatial_norm = nn.BatchNorm2d(grouped_channels)
        
        # ★ 新增：用 Linear 取代原本的暴力 mean()，保留臉部空間結構
        self.face_proj = nn.Linear(grouped_channels * num_keypoints, grouped_channels)
        
        # 2. Temporal
        self.temporal = nn.Conv1d(
            in_channels=grouped_channels,
            out_channels=temporal_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            bias=False,
        )
        self.temporal_norm = nn.BatchNorm1d(temporal_channels)
        
        # 3. LSTM & Attention
        self.lstm = nn.LSTM(
            input_size=temporal_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(
        self, face_landmarks: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # face_landmarks: (B, T, K, 3)
        B, T, K, C = face_landmarks.shape
        
        # ==========================================
        # ★ 必殺技 1：鼻尖正規化 (Nose Normalization)
        # 假設 Index 0 或 1 是鼻子中心，以此為相對原點
        # ==========================================
        nose = face_landmarks[:, :, 0:1, :] 
        face_norm = face_landmarks - nose
        
        # 用正規化後的臉部算 delta
        face_delta = torch.zeros_like(face_norm)
        face_delta[:, 1:] = face_norm[:, 1:] - face_norm[:, :-1]
        
        # 結合起來 (B, T, K, 6)
        face_input = torch.cat([face_norm, face_delta], dim=-1) 

        # ==========================================
        # ★ 必殺技 2：保留臉部拓樸，拒絕泥巴化
        # ==========================================
        x = face_input.permute(0, 3, 1, 2)  # (B, 6, T, K)
        x = F.relu(self.spatial_norm(self.spatial(x))) # (B, grouped_channels, T, K)
        
        # 把 Channel 和 Keypoints 攤平：(B, T, grouped_channels * K)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1) 
        
        # 投影回 (B, T, grouped_channels)，這樣模型就學會了「整張臉的組合表情」
        x = self.face_proj(x)
        
        # 轉成 Conv1d 需要的 (B, C, T)
        x = x.permute(0, 2, 1)

        # 剩下的 Temporal 與 LSTM 保持原樣
        x = F.relu(self.temporal_norm(self.temporal(x)))  # (B, temporal_channels, T)
        x = x.permute(0, 2, 1)  # (B, T, temporal_channels)
        seq, _ = self.lstm(x)  # (B, T, 256)
        seq = self.lstm_dropout(seq)

        attn_logits = self.attn(seq)
        if valid_mask is not None:
            attn_logits = attn_logits.masked_fill(~valid_mask.bool().unsqueeze(-1), -1e4)
        attn_w = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        attn_w = self.attn_dropout(attn_w)
        pooled = torch.sum(attn_w * seq, dim=1)  # (B, 256)
        return pooled, seq
