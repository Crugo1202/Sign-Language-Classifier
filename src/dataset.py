from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import KEYPOINT_DIR, METADATA_PATH


@dataclass
class SequenceExample:
    path: Path
    label: int
    length: int


class WLASLDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path = METADATA_PATH,
        keypoint_dir: Path = KEYPOINT_DIR,
        split: str = "train",
        gloss_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.keypoint_dir = Path(keypoint_dir)
        self.split = split
        self.transform = transform

        self.samples: List[SequenceExample] = []
        self.gloss_to_idx: Dict[str, int] = gloss_to_idx or {}

        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != self.split:
                    continue

                gloss = row["gloss"]
                if gloss not in self.gloss_to_idx:
                    self.gloss_to_idx[gloss] = len(self.gloss_to_idx)

                label = self.gloss_to_idx[gloss]
                filename = row["filename"]
                path = self.keypoint_dir / filename
                if not path.exists():
                    continue

                length = int(np.load(path, mmap_mode="r").shape[0])
                self.samples.append(SequenceExample(path=path, label=label, length=length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        seq = np.load(sample.path).astype(np.float32)

        if self.transform is not None:
            seq = self.transform(seq)

        x = torch.from_numpy(seq)  # (T, D)
        length = x.shape[0]
        label = sample.label
        return x, length, label


def compute_normalization_stats(
    dataset: WLASLDataset,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and std over the (train) dataset.
    """
    sums = None
    sq_sums = None
    count = 0

    for i, (x, _, _) in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        arr = x.numpy()  # (T, D)
        if sums is None:
            sums = arr.sum(axis=(0, 0))
            sq_sums = (arr ** 2).sum(axis=(0, 0))
        else:
            sums += arr.sum(axis=(0, 0))
            sq_sums += (arr ** 2).sum(axis=(0, 0))
        count += arr.shape[0]

    if count == 0:
        raise ValueError("No samples to compute normalization statistics.")

    mean = sums / count
    var = sq_sums / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class Standardize:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    Returns:
      padded_sequences: (batch, max_len, D)
      lengths: (batch,)
      labels: (batch,)
    """
    sequences, lengths, labels = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded, lengths_tensor, labels_tensor

