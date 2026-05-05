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
        grouped_channels: int = 16,
        temporal_channels: int = 128,
        temporal_kernel: int = 5,
        hidden_size: int = 128,
    ):
        super().__init__()
        # face + delta => 6 channels, then compress keypoint interactions
        self.spatial = nn.Conv2d(
            in_channels=6, out_channels=grouped_channels, kernel_size=1, stride=1, bias=False
        )
        self.spatial_norm = nn.BatchNorm2d(grouped_channels)
        self.temporal = nn.Conv1d(
            in_channels=grouped_channels,
            out_channels=temporal_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            bias=False,
        )
        self.temporal_norm = nn.BatchNorm1d(temporal_channels)
        self.lstm = nn.LSTM(
            input_size=temporal_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(
        self, face_landmarks: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # face_landmarks: (B, T, K, 3)
        face_delta = torch.zeros_like(face_landmarks)
        face_delta[:, 1:] = face_landmarks[:, 1:] - face_landmarks[:, :-1]
        face_input = torch.cat([face_landmarks, face_delta], dim=-1)  # (B, T, K, 6)

        x = face_input.permute(0, 3, 1, 2)  # (B, 6, T, K)
        x = F.relu(self.spatial_norm(self.spatial(x)))
        x = x.mean(dim=-1)  # (B, grouped_channels, T)

        x = F.relu(self.temporal_norm(self.temporal(x)))  # (B, temporal_channels, T)
        x = x.permute(0, 2, 1)  # (B, T, temporal_channels)
        seq, _ = self.lstm(x)  # (B, T, 256)

        attn_logits = self.attn(seq)
        if valid_mask is not None:
            attn_logits = attn_logits.masked_fill(~valid_mask.bool().unsqueeze(-1), -1e4)
        attn_w = F.softmax(attn_logits, dim=1)  # (B, T, 1)
        pooled = torch.sum(attn_w * seq, dim=1)  # (B, 256)
        return pooled, seq
