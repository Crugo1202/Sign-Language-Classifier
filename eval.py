from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from models import DSLNetUpgraded, DSLNetUpgradedLoss
from train import WLASLSkeletonDataset, move_batch_to_device, topk_accuracy


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: DSLNetUpgradedLoss,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0}
    steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("label").long()
        logits, aux = model(batch)
        loss = criterion(logits, labels, aux)["total"]
        acc = topk_accuracy(logits, labels, ks=(1, 5, 10))

        totals["loss"] += loss.item()
        totals["top1"] += acc["top1"]
        totals["top5"] += acc["top5"]
        totals["top10"] += acc["top10"]
        steps += 1

    return {k: v / max(steps, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    split_dir = Path(data_cfg["root"]) / data_cfg["test_split"]
    ds = WLASLSkeletonDataset(
        split_dir,
        t_max=data_cfg["t_max"],
        training=False,
        preprocess_cfg=cfg.get("preprocess", {}),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    model = DSLNetUpgraded(num_classes=cfg["num_classes"], cfg=cfg["model"]).to(device)
    criterion = DSLNetUpgradedLoss(
        alpha_geo=cfg["train"]["alpha_geo"],
        beta_sym=cfg["train"]["beta_sym"],
        label_smoothing=cfg["train"]["label_smoothing"],
        feature_dim=cfg["model"]["fusion"]["shared_dim"],
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, criterion, loader, device)

    print(
        f"Test metrics | loss={metrics['loss']:.4f} "
        f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} top10={metrics['top10']:.4f}"
    )


if __name__ == "__main__":
    main()
