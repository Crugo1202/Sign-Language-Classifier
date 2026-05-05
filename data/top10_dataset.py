from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .landmark_layout import split_holistic_landmarks
from .preprocess import PreprocessConfig, preprocess_sample


class TopKGlossLandmarksDataset(Dataset):
    """
    Dataset backed by:
      - one big holistic landmark npz keyed by sample index string
      - WLASL_parsed_data.json carrying gloss labels and split metadata
    """

    def __init__(
        self,
        landmarks_npz_path: str,
        parsed_json_path: str,
        sample_indices: Sequence[int],
        label_map: Dict[str, int],
        t_max: int = 64,
        training: bool = False,
        preprocess_cfg: Dict[str, object] | None = None,
    ):
        self.landmarks_npz_path = str(landmarks_npz_path)
        self.parsed_json_path = str(parsed_json_path)
        self.sample_indices = [int(i) for i in sample_indices]
        self.label_map = dict(label_map)
        cfg_dict = {"t_max": t_max}
        if preprocess_cfg:
            cfg_dict.update(preprocess_cfg)
        self.cfg = PreprocessConfig(**cfg_dict)
        self.training = training

        with open(self.parsed_json_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.labels = [int(self.label_map[self.records[i]["gloss"]]) for i in self.sample_indices]
        self._npz = None
        self._ram_cache: Dict[int, np.ndarray] | None = None
        self._tensor_cache: Dict[int, Dict[str, torch.Tensor]] | None = None

        self._preload_to_ram()
        if not self.training:
            self._prebuild_eval_tensor_cache()

    def _ensure_open(self) -> np.lib.npyio.NpzFile:
        if self._npz is None:
            self._npz = np.load(self.landmarks_npz_path, allow_pickle=False)
        return self._npz

    def _preload_to_ram(self) -> None:
        npz = self._ensure_open()
        self._ram_cache = {}
        for sample_idx in self.sample_indices:
            key = str(sample_idx)
            if key not in npz.files:
                raise KeyError(f"Key '{key}' not found in landmarks npz.")
            self._ram_cache[int(sample_idx)] = npz[key].astype(np.float32)

    def _prebuild_eval_tensor_cache(self) -> None:
        if self._ram_cache is None:
            raise RuntimeError("RAM cache is not initialized.")
        self._tensor_cache = {}
        for sample_idx in self.sample_indices:
            seq = self._ram_cache[int(sample_idx)]
            streams = split_holistic_landmarks(seq)
            pre = preprocess_sample(streams, self.cfg, training=False)
            record = self.records[int(sample_idx)]
            pre["label"] = np.int64(self.label_map[record["gloss"]])
            out: Dict[str, torch.Tensor] = {}
            for k, v in pre.items():
                if isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(v)
                else:
                    out[k] = torch.tensor(v)
            self._tensor_cache[int(sample_idx)] = out

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.sample_indices[idx]
        if self._tensor_cache is not None:
            return self._tensor_cache[int(sample_idx)]

        record = self.records[sample_idx]
        gloss = record["gloss"]
        if gloss not in self.label_map:
            raise KeyError(f"Gloss '{gloss}' is not in label_map.")

        if self._ram_cache is None:
            raise RuntimeError("RAM cache is not initialized.")
        if sample_idx not in self._ram_cache:
            raise KeyError(f"Sample index '{sample_idx}' not found in RAM cache.")
        seq = self._ram_cache[sample_idx]  # (T, V, 3)
        streams = split_holistic_landmarks(seq)
        pre = preprocess_sample(streams, self.cfg, training=self.training)
        pre["label"] = np.int64(self.label_map[gloss])
        out: Dict[str, torch.Tensor] = {}
        for k, v in pre.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
            else:
                out[k] = torch.tensor(v)
        return out


def build_topk_gloss_index(
    parsed_json_path: str,
    top_k: int = 10,
) -> Dict[str, object]:
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, object]] = json.load(f)

    counts = Counter(r["gloss"] for r in records)
    top_items = counts.most_common(top_k)
    top_glosses = [g for g, _ in top_items]
    label_map = {g: i for i, g in enumerate(top_glosses)}

    kept_indices: List[int] = []
    for i, rec in enumerate(records):
        if rec["gloss"] in label_map:
            kept_indices.append(i)

    return {
        "label_map": label_map,
        "top_glosses": top_glosses,
        "counts": dict(top_items),
        "indices": kept_indices,
    }


def build_topk_gloss_split_index(
    parsed_json_path: str,
    top_k: int = 10,
) -> Dict[str, object]:
    meta = build_topk_gloss_index(parsed_json_path=parsed_json_path, top_k=top_k)
    label_map = meta["label_map"]
    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    with open(parsed_json_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, object]] = json.load(f)

    for i, rec in enumerate(records):
        gloss = rec["gloss"]
        if gloss not in label_map:
            continue
        split = str(rec.get("split", "")).lower()
        if split in split_indices:
            split_indices[split].append(i)

    return {
        "label_map": label_map,
        "top_glosses": meta["top_glosses"],
        "counts": meta["counts"],
        "split_indices": split_indices,
    }


def stratified_kfold_indices(
    sample_indices: Sequence[int],
    labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    if len(sample_indices) != len(labels):
        raise ValueError("sample_indices and labels must have same length.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = {}
    for sid, y in zip(sample_indices, labels):
        by_label.setdefault(int(y), []).append(int(sid))

    fold_buckets: List[List[int]] = [[] for _ in range(n_splits)]
    for y in sorted(by_label.keys()):
        cls_ids = by_label[y]
        rng.shuffle(cls_ids)
        for i, sid in enumerate(cls_ids):
            fold_buckets[i % n_splits].append(sid)

    all_ids = set(int(x) for x in sample_indices)
    folds: List[Dict[str, List[int]]] = []
    for test_ids in fold_buckets:
        test_set = set(test_ids)
        train_ids = sorted(all_ids - test_set)
        folds.append({"train": train_ids, "test": sorted(test_set)})
    return folds
