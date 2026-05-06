from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.preprocess import PreprocessConfig, apply_feature_augmentation, preprocess_sample
from models import DSLNetUpgraded, DSLNetUpgradedLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_valid_mask(sample: Dict[str, np.ndarray], t_max: int) -> Tuple[np.ndarray, int]:
    if "valid_mask" in sample:
        mask = sample["valid_mask"].astype(np.bool_)
        valid_len = int(mask.sum())
        return mask, valid_len
    if "valid_length" in sample:
        valid_len = int(np.clip(sample["valid_length"], 0, t_max))
        mask = np.zeros((t_max,), dtype=np.bool_)
        mask[:valid_len] = True
        return mask, valid_len

    # Fallback for preprocessed NPZ without explicit mask metadata.
    signal = (
        np.abs(sample["right_hand_shape"]).sum(axis=(1, 2))
        + np.abs(sample["left_hand_shape"]).sum(axis=(1, 2))
        + np.abs(sample["dual_wrist_traj"]).sum(axis=1)
        + np.abs(sample["face_landmarks_norm"]).sum(axis=(1, 2))
    )
    mask = signal > 1e-6
    if not mask.any():
        mask[:] = True
    valid_len = int(mask.sum())
    return mask.astype(np.bool_), valid_len


class WLASLSkeletonDataset(Dataset):
    """
    Expected NPZ format per sample:
      EITHER preprocessed:
        - right_hand_shape: (T, 21, 3)
        - left_hand_shape: (T, 21, 3)
        - dual_wrist_traj: (T, 6)
        - face_landmarks_norm: (T, K, 3), K is semantic facial keypoints (e.g. 60-80)
      OR raw:
        - right_hand: (T, 21, 3)
        - left_hand: (T, 21, 3)
        - face: (T, 468/478, 3) or selected keypoints (T, K, 3)
        - pose: (T, 33, 3)
      Required label key:
        - label: int
    """

    def __init__(
        self,
        split_dir: Path,
        t_max: int,
        training: bool,
        preprocess_cfg: Dict[str, object] | None = None,
    ):
        self.files = sorted(split_dir.glob("*.npz"))
        cfg_dict = {"t_max": t_max}
        if preprocess_cfg:
            cfg_dict.update(preprocess_cfg)
        self.cfg = PreprocessConfig(**cfg_dict)
        self.training = training
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = np.load(self.files[idx], allow_pickle=False)
        keys = set(item.files)

        if {"right_hand_shape", "left_hand_shape", "dual_wrist_traj", "face_landmarks_norm"}.issubset(keys):
            sample = {
                "right_hand_shape": item["right_hand_shape"].astype(np.float32),
                "left_hand_shape": item["left_hand_shape"].astype(np.float32),
                "dual_wrist_traj": item["dual_wrist_traj"].astype(np.float32),
                "face_landmarks_norm": item["face_landmarks_norm"].astype(np.float32),
            }
            if "hand_orientation" in keys:
                sample["hand_orientation"] = item["hand_orientation"].astype(np.float32)
            if "valid_mask" in keys:
                sample["valid_mask"] = item["valid_mask"].astype(np.bool_)
            if "valid_length" in keys:
                sample["valid_length"] = np.int64(item["valid_length"])
            if self.training:
                sample = apply_feature_augmentation(sample, self.cfg)
        else:
            raw = {
                "right_hand": item["right_hand"].astype(np.float32),
                "left_hand": item["left_hand"].astype(np.float32),
                "face": item["face"].astype(np.float32),
                "pose": item["pose"].astype(np.float32),
            }
            sample = preprocess_sample(raw, self.cfg, training=self.training)

        valid_mask, valid_len = infer_valid_mask(sample, self.cfg.t_max)
        sample["valid_mask"] = valid_mask
        sample["valid_length"] = np.int64(valid_len)

        label = int(item["label"])
        sample["label"] = np.int64(label)
        out: Dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
            else:
                out[k] = torch.tensor(v)
        return out


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    max_k = min(max(ks), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        k_eff = min(k, logits.size(1))
        correct_k = correct[:k_eff].reshape(-1).float().sum()
        out[f"top{k}"] = (correct_k / labels.size(0)).item()
    return out


def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    root = Path(data_cfg["root"])
    preprocess_cfg = cfg.get("preprocess", {})
    train_ds = WLASLSkeletonDataset(
        root / data_cfg["train_split"],
        data_cfg["t_max"],
        training=True,
        preprocess_cfg=preprocess_cfg,
    )
    val_ds = WLASLSkeletonDataset(
        root / data_cfg["val_split"],
        data_cfg["t_max"],
        training=False,
        preprocess_cfg=preprocess_cfg,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def train_one_epoch(
    model: nn.Module,
    criterion: DSLNetUpgradedLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("label").long()

        logits, aux = model(batch)
        loss_dict = criterion(logits, labels, aux)
        loss = loss_dict["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        acc = topk_accuracy(logits.detach(), labels)
        running["loss"] += loss.item()
        running["top1"] += acc["top1"]
        running["top5"] += acc["top5"]
        running["top10"] += acc["top10"]
        steps += 1

    return {k: v / max(steps, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: DSLNetUpgradedLoss,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
    steps = 0

    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("label").long()

        logits, aux = model(batch)
        loss = criterion(logits, labels, aux)["total"]
        acc = topk_accuracy(logits, labels)

        running["loss"] += loss.item()
        running["top1"] += acc["top1"]
        running["top5"] += acc["top5"]
        running["top10"] += acc["top10"]
        steps += 1

    return {k: v / max(steps, 1) for k, v in running.items()}


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model = DSLNetUpgraded(num_classes=cfg["num_classes"], cfg=cfg["model"]).to(device)
    criterion = DSLNetUpgradedLoss(
        alpha_geo=cfg["train"]["alpha_geo"],
        beta_sym=cfg["train"]["beta_sym"],
        label_smoothing=cfg["train"]["label_smoothing"],
        feature_dim=cfg["model"]["fusion"]["shared_dim"],
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"]["warmup_epochs"]
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    global_step = 0

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["train"]["save_name"]

    best_top1 = 0.0
    for epoch in range(1, epochs + 1):
        for pg in optimizer.param_groups:
            pg["lr"] = cfg["train"]["learning_rate"] * cosine_with_warmup(
                global_step, total_steps, warmup_steps
            )

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg["train"]["grad_clip"],
        )
        global_step += len(train_loader)

        val_metrics = evaluate(model=model, criterion=criterion, loader=val_loader, device=device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
            f"val_top5={val_metrics['top5']:.4f} val_top10={val_metrics['top10']:.4f}"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path} (best top1={best_top1:.4f})")


if __name__ == "__main__":
    main()
