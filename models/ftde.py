from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        return y


class FTDEDual(nn.Module):
    """
    Finsler Trajectory Dynamics Encoder for dual-wrist trajectories.
    ★ 升級版：直接接收完整 Pose (33點)，自動進行「胸腔中心正規化」
    Input: (B, T, 33, 3) 
    """
    def __init__(
        self,
        conv_channels: int = 64,
        conv_kernel: int = 7,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        alpha_finsler: float = 1.5,
        temperature_tau: float = 0.5,
        dropout: float = 0.0,
        attn_dropout: float | None = None,
    ):
        super().__init__()
        self.alpha_finsler = alpha_finsler
        self.temperature_tau = temperature_tau
        attn_dropout = dropout if attn_dropout is None else attn_dropout

        self.causal_conv = CausalConv1d(10, conv_channels, kernel_size=conv_kernel)
        self.norm = nn.BatchNorm1d(conv_channels)
        self.act = nn.ReLU(inplace=True)
        self.temporal_model = nn.LSTM(
            input_size=conv_channels, hidden_size=hidden_size, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.phi = nn.Sequential(
            nn.Linear(13, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Softplus(),
        )

    def _finsler_energy(self, traj: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        vel = torch.zeros_like(traj)
        vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        direction = vel / (speed + 1e-6)

        right_vel, left_vel = vel[..., :3], vel[..., 3:]
        sync = 1.0 / (1.0 + torch.linalg.norm(right_vel - left_vel, dim=-1, keepdim=True))

        phi_input = torch.cat([traj, direction, sync], dim=-1)
        phi = self.phi(phi_input)
        energy = phi * torch.pow(speed + 1e-6, self.alpha_finsler)
        logits = energy / self.temperature_tau
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.bool().unsqueeze(-1), -1e4)
        weights = F.softmax(logits, dim=1)
        weights = self.attn_dropout(weights)
        return weights

    def forward(
        self, pose_landmarks: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pose_landmarks 預期形狀: (B, T, 33, 3)
        
        # ==========================================
        # ★ 必殺技：胸腔中心正規化 (Neck Normalization)
        # 抓取 MediaPipe Pose 點位：
        # 11: 左肩 (Left Shoulder), 12: 右肩 (Right Shoulder)
        # 15: 左手腕 (Left Wrist), 16: 右手腕 (Right Wrist)
        # ==========================================
        left_shoulder = pose_landmarks[:, :, 11, :]
        right_shoulder = pose_landmarks[:, :, 12, :]
        neck = (left_shoulder + right_shoulder) / 2.0  # 計算肩膀中心點 (脖子/胸腔)

        raw_left_wrist = pose_landmarks[:, :, 15, :]
        raw_right_wrist = pose_landmarks[:, :, 16, :]

        # 將手腕座標減去脖子座標 (獲得以身體為中心的相對軌跡)
        norm_left_wrist = raw_left_wrist - neck
        norm_right_wrist = raw_right_wrist - neck

        # 組合出你原本需要的 dual_wrist_traj (B, T, 6)
        # 注意順序：你原本代碼是 right_vel = vel[..., :3] 所以 Right 在前
        dual_wrist_traj = torch.cat([norm_right_wrist, norm_left_wrist], dim=-1)
        
        # ==========================================
        # 以下完美沿用你原本強大的 Finsler 計算邏輯
        # ==========================================
        right, left = dual_wrist_traj[..., :3], dual_wrist_traj[..., 3:]
        relative = right - left

        vel = torch.zeros_like(dual_wrist_traj)
        vel[:, 1:] = dual_wrist_traj[:, 1:] - dual_wrist_traj[:, :-1]
        sync = 1.0 / (
            1.0 + torch.linalg.norm(vel[..., :3] - vel[..., 3:], dim=-1, keepdim=True)
        )
        
        # (B, T, 10) -> (B, C, T) -> Conv -> (B, T, C)
        conv_input = torch.cat([dual_wrist_traj, relative, sync], dim=-1) 
        x = conv_input.permute(0, 2, 1)  
        x = self.act(self.norm(self.causal_conv(x)))
        x = x.permute(0, 2, 1)  

        seq, _ = self.temporal_model(x)  # (B, T, 512)
        seq = self.lstm_dropout(seq)
        
        # 使用正規化後的軌跡來計算 Finsler Energy (過濾靜止畫面)
        weights = self._finsler_energy(dual_wrist_traj, valid_mask=valid_mask)  # (B, T, 1)
        pooled = torch.sum(weights * seq, dim=1)
        
        return pooled, seq
