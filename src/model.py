from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (
    BIDIRECTIONAL,
    DROPOUT,
    EMBED_DIM,
    HIDDEN_SIZE,
    INPUT_DIM,
    NUM_GLOSSES,
    NUM_LAYERS,
)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        outputs: (batch, seq_len, hidden)
        lengths: (batch,)
        """
        energy = torch.tanh(self.attn(outputs))  # (B, T, H)
        scores = self.v(energy).squeeze(-1)  # (B, T)

        # Mask out padding
        max_len = outputs.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # (B, H)
        return context


class SignLanguageLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: Optional[int] = NUM_GLOSSES,
        dropout: float = DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
    ) -> None:
        super().__init__()

        if num_classes is None:
            raise ValueError("num_classes must be specified (NUM_GLOSSES in config.py).")

        self.proj = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.pool = AttentionPooling(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_dim)
        lengths: (batch,)
        """
        embedded = self.proj(x)

        # Pack sequences for efficient LSTM processing
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        pooled = self.pool(outputs, lengths)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits, pooled

