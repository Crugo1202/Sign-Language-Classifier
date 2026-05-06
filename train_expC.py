from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.top10_dataset import TopKGlossLandmarksDataset, build_topk_gloss_split_index
from models.model_pair import TFTSignPairwise


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
    steps = 0
    for batch in tqdm(loader, desc="train" if training else "eval", leave=False, disable=not show_progress):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("label").long()
        logits, _ = model(batch)
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        acc = topk_accuracy(logits.detach(), labels)
        running["loss"] += float(loss.item())
        running["top1"] += acc["top1"]
        running["top5"] += acc["top5"]
        running["top10"] += acc["top10"]
        steps += 1
    return {k: v / max(1, steps) for k, v in running.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--landmarks-npz", type=str, default="data/skeleton/landmarks_V3.npz")
    parser.add_argument("--parsed-json", type=str, default="data/WLASL_parsed_data.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default="checkpoints/topk_official_tft.json")
    parser.add_argument("--save-ckpt", type=str, default="checkpoints/topk_official_tft_best.pt")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def apply_config_defaults(args: argparse.Namespace, cfg: Dict) -> argparse.Namespace:
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    args.epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 120)
    args.patience = args.patience if args.patience is not None else train_cfg.get("patience", 20)
    args.batch_size = (
        args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 16)
    )
    args.t_max = args.t_max if args.t_max is not None else data_cfg.get("t_max", 64)
    args.lr = args.lr if args.lr is not None else train_cfg.get("learning_rate", 1e-4)
    args.weight_decay = (
        args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-4)
    )
    args.label_smoothing = (
        args.label_smoothing
        if args.label_smoothing is not None
        else train_cfg.get("label_smoothing", 0.0)
    )
    return args


def build_balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weights[np.asarray(labels, dtype=np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = apply_config_defaults(args, cfg)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    preprocess_cfg = cfg.get("preprocess", {})
    train_cfg = cfg.get("train", {})

    meta = build_topk_gloss_split_index(args.parsed_json, top_k=args.top_k)
    label_map = meta["label_map"]
    split_indices = meta["split_indices"]
    train_ids = split_indices["train"]
    val_ids = split_indices["val"]
    test_ids = split_indices["test"]
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise RuntimeError("Official split has empty train/test after top-k filtering.")
    if len(val_ids) == 0:
        raise RuntimeError("Official val split is empty; cannot do early stopping.")

    train_ds = TopKGlossLandmarksDataset(
        args.landmarks_npz,
        args.parsed_json,
        train_ids,
        label_map,
        t_max=args.t_max,
        training=True,
        preprocess_cfg=preprocess_cfg,
    )
    val_ds = TopKGlossLandmarksDataset(
        args.landmarks_npz,
        args.parsed_json,
        val_ids,
        label_map,
        t_max=args.t_max,
        training=False,
        preprocess_cfg=preprocess_cfg,
    )
    test_ds = TopKGlossLandmarksDataset(
        args.landmarks_npz,
        args.parsed_json,
        test_ids,
        label_map,
        t_max=args.t_max,
        training=False,
        preprocess_cfg=preprocess_cfg,
    )
    balanced_sampler = build_balanced_sampler(train_ds.labels) if train_cfg.get("balanced_sampling", False) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=balanced_sampler is None,
        sampler=balanced_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TFTSignPairwise(num_classes=args.top_k, cfg=cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.05, 1e-6))

    best_val_top1 = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, show_progress=not args.no_progress
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device, show_progress=not args.no_progress
            )
        scheduler.step()

        print(
            f"[epoch {epoch:03d}] "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"train_top1={train_metrics['top1']:.4f} "
            f"val_top1={val_metrics['top1']:.4f}",
            flush=True,
        )
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint state.")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, criterion, device, show_progress=not args.no_progress)

    Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "best_val_top1": best_val_top1,
            "best_epoch": best_epoch,
            "label_map": label_map,
            "config": cfg,
        },
        args.save_ckpt,
    )
    summary = {
        "model_name": "tft",
        "top_k": args.top_k,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_top10": test_metrics["top10"],
        "top_glosses": meta["top_glosses"],
        "counts": meta["counts"],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved checkpoint to {args.save_ckpt}", flush=True)
    print(f"Saved summary to {args.output_json}", flush=True)
    print(
        f"best_epoch={best_epoch} best_val_top1={best_val_top1:.4f} "
        f"test_top1={summary['test_top1']:.4f} test_top5={summary['test_top5']:.4f} "
        f"test_top10={summary['test_top10']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
