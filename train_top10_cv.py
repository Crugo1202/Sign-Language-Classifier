from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.top10_dataset import TopKGlossLandmarksDataset, build_topk_gloss_index, stratified_kfold_indices
from models import DSLNetUpgraded, TFTSignLite


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
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def forward_logits(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(batch)
    if isinstance(out, tuple):
        return out[0]
    return out


@dataclass
class FoldResult:
    fold_idx: int
    best_val_top1: float
    test_top1: float
    test_top5: float
    test_top10: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
    steps = 0
    for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("label").long()
        logits = forward_logits(model, batch)
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
    return {k: v / max(steps, 1) for k, v in running.items()}


def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return total ** 0.5


def run_single_batch_overfit(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    steps: int,
) -> Dict[str, float]:
    model.train()
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    labels = batch.pop("label").long()

    final_metrics: Dict[str, float] = {}
    for step in range(1, steps + 1):
        logits = forward_logits(model, batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_global_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = topk_accuracy(logits.detach(), labels, ks=(1, 5, 10))
        logit_min = float(logits.detach().min().item())
        logit_max = float(logits.detach().max().item())
        final_metrics = {
            "loss": float(loss.item()),
            "top1": acc["top1"],
            "top5": acc["top5"],
            "top10": acc["top10"],
            "grad_norm": gnorm,
            "logit_min": logit_min,
            "logit_max": logit_max,
        }
        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"[single-batch step {step:03d}/{steps}] "
                f"loss={final_metrics['loss']:.4f} top1={final_metrics['top1']:.4f} "
                f"top5={final_metrics['top5']:.4f} top10={final_metrics['top10']:.4f} "
                f"grad_norm={final_metrics['grad_norm']:.4f} "
                f"logits=[{final_metrics['logit_min']:.4f}, {final_metrics['logit_max']:.4f}]"
            )
    return final_metrics


def build_model(model_name: str, cfg: Dict, num_classes: int) -> nn.Module:
    if model_name == "dslnet":
        return DSLNetUpgraded(num_classes=num_classes, cfg=cfg["model"])
    if model_name == "tft":
        return TFTSignLite(num_classes=num_classes, cfg=cfg["model"])
    raise ValueError(f"Unsupported model_name={model_name}")


def split_train_val(ids: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    ids = np.array(list(ids), dtype=np.int64)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_ratio)))
    val_ids = ids[:n_val].tolist()
    train_ids = ids[n_val:].tolist()
    return train_ids, val_ids


def run_kfold(args: argparse.Namespace) -> Dict[str, object]:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    topk_meta = build_topk_gloss_index(args.parsed_json, top_k=args.top_k)
    label_map = topk_meta["label_map"]
    sample_indices = topk_meta["indices"]

    with open(args.parsed_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    labels = [label_map[records[i]["gloss"]] for i in sample_indices]

    folds = stratified_kfold_indices(
        sample_indices=sample_indices,
        labels=labels,
        n_splits=args.folds,
        seed=args.seed,
    )

    overfit_label_smoothing = 0.0 if args.single_batch_overfit else args.label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=overfit_label_smoothing)
    preprocess_cfg = cfg.get("preprocess", {})
    fold_results: List[FoldResult] = []
    for fi, fold in enumerate(folds, start=1):
        train_ids_full = fold["train"]
        test_ids = fold["test"]
        train_ids, val_ids = split_train_val(train_ids_full, val_ratio=args.val_ratio, seed=args.seed + fi)

        train_ds = TopKGlossLandmarksDataset(
            landmarks_npz_path=args.landmarks_npz,
            parsed_json_path=args.parsed_json,
            sample_indices=train_ids,
            label_map=label_map,
            t_max=args.t_max,
            training=True,
            preprocess_cfg=preprocess_cfg,
        )
        val_ds = TopKGlossLandmarksDataset(
            landmarks_npz_path=args.landmarks_npz,
            parsed_json_path=args.parsed_json,
            sample_indices=val_ids,
            label_map=label_map,
            t_max=args.t_max,
            training=False,
            preprocess_cfg=preprocess_cfg,
        )
        test_ds = TopKGlossLandmarksDataset(
            landmarks_npz_path=args.landmarks_npz,
            parsed_json_path=args.parsed_json,
            sample_indices=test_ids,
            label_map=label_map,
            t_max=args.t_max,
            training=False,
            preprocess_cfg=preprocess_cfg,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
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

        model = build_model(args.model_name, cfg=cfg, num_classes=args.top_k).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.single_batch_overfit:
            overfit_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            final_metrics = run_single_batch_overfit(
                model=model,
                loader=overfit_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                steps=args.overfit_steps,
            )
            summary = {
                "mode": "single_batch_overfit",
                "model_name": args.model_name,
                "top_k": args.top_k,
                "fold": fi,
                "steps": args.overfit_steps,
                "final_metrics": final_metrics,
            }
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to {args.output_json}")
            return summary

        best_state = None
        best_val_top1 = -1.0
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, criterion, device)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, criterion, device)
            if val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[fold {fi}/{args.folds}] epoch {epoch:03d} "
                f"train_top1={train_metrics['top1']:.4f} val_top1={val_metrics['top1']:.4f}"
            )

        if best_state is not None:
            model.load_state_dict(best_state)
        with torch.no_grad():
            test_metrics = run_epoch(model, test_loader, None, criterion, device)
        fold_results.append(
            FoldResult(
                fold_idx=fi,
                best_val_top1=best_val_top1,
                test_top1=test_metrics["top1"],
                test_top5=test_metrics["top5"],
                test_top10=test_metrics["top10"],
            )
        )
        print(
            f"[fold {fi}] best_val_top1={best_val_top1:.4f} "
            f"test_top1={test_metrics['top1']:.4f} "
            f"test_top5={test_metrics['top5']:.4f} "
            f"test_top10={test_metrics['top10']:.4f}"
        )

    summary = {
        "model_name": args.model_name,
        "top_k": args.top_k,
        "folds": args.folds,
        "epochs": args.epochs,
        "fold_results": [fr.__dict__ for fr in fold_results],
        "mean_test_top1": mean([fr.test_top1 for fr in fold_results]),
        "std_test_top1": pstdev([fr.test_top1 for fr in fold_results]),
        "mean_test_top5": mean([fr.test_top5 for fr in fold_results]),
        "mean_test_top10": mean([fr.test_top10 for fr in fold_results]),
        "top_glosses": topk_meta["top_glosses"],
        "counts": topk_meta["counts"],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {args.output_json}")
    print(
        f"mean_test_top1={summary['mean_test_top1']:.4f} "
        f"std_test_top1={summary['std_test_top1']:.4f} "
        f"mean_test_top5={summary['mean_test_top5']:.4f} "
        f"mean_test_top10={summary['mean_test_top10']:.4f}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--landmarks-npz", type=str, default="data/skeleton/landmarks_V3.npz")
    parser.add_argument("--parsed-json", type=str, default="data/WLASL_parsed_data.json")
    parser.add_argument("--model-name", type=str, choices=["dslnet", "tft"], default="tft")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--t-max", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default="checkpoints/top10_cv_summary.json")
    parser.add_argument("--single-batch-overfit", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run_kfold(parse_args())
