import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from config import BEST_MODEL_PATH, KEYPOINT_DIR, METADATA_PATH, NUM_WORKERS
from dataset import WLASLDataset, Standardize, collate_fn, compute_normalization_stats
from model import SignLanguageLSTM


def load_model(
    checkpoint_path: Path,
    num_classes: int,
) -> Tuple[SignLanguageLSTM, Dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    gloss_to_idx = ckpt["gloss_to_idx"]

    model = SignLanguageLSTM(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    return model, gloss_to_idx


def build_test_dataset(
    metadata_path: Path,
    keypoint_dir: Path,
    gloss_to_idx: Dict[str, int],
) -> WLASLDataset:
    # For normalization, recompute on train split using shared mapping
    train_ds = WLASLDataset(
        metadata_path=metadata_path,
        keypoint_dir=keypoint_dir,
        split="train",
        gloss_to_idx=gloss_to_idx,
        transform=None,
    )
    mean, std = compute_normalization_stats(train_ds)
    transform = Standardize(mean, std)

    test_ds = WLASLDataset(
        metadata_path=metadata_path,
        keypoint_dir=keypoint_dir,
        split="test",
        gloss_to_idx=gloss_to_idx,
        transform=transform,
    )
    return test_ds


def evaluate(
    checkpoint_path: Path,
    metadata_path: Path,
    keypoint_dir: Path,
    num_workers: int = NUM_WORKERS,
    metrics_out: Path = Path("metrics.json"),
) -> None:
    # Load mapping and model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    gloss_to_idx: Dict[str, int] = ckpt["gloss_to_idx"]
    idx_to_gloss: List[str] = [None] * len(gloss_to_idx)
    for g, i in gloss_to_idx.items():
        idx_to_gloss[i] = g

    num_classes = len(gloss_to_idx)
    model, _ = load_model(checkpoint_path, num_classes=num_classes)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_ds = build_test_dataset(
        metadata_path=metadata_path,
        keypoint_dir=keypoint_dir,
        gloss_to_idx=gloss_to_idx,
    )
    loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for x, lengths, labels in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits, _ = model(x, lengths)
            preds = logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    if not all_labels:
        raise ValueError("No test samples found to evaluate.")

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    per_class_acc = {}
    cm_diag = np.diag(cm)
    cm_sum = cm.sum(axis=1)
    for i, gloss in enumerate(idx_to_gloss):
        if cm_sum[i] > 0:
            per_class_acc[gloss] = float(cm_diag[i] / cm_sum[i])
        else:
            per_class_acc[gloss] = None

    metrics = {
        "accuracy": float(acc),
        "per_class_accuracy": per_class_acc,
    }

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save("confusion_matrix.npy", cm)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Saved metrics to {metrics_out} and confusion_matrix.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(METADATA_PATH),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default=str(KEYPOINT_DIR),
        help="Directory containing keypoint .npy files.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="metrics.json",
        help="Output JSON file for metrics.",
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        metadata_path=Path(args.metadata),
        keypoint_dir=Path(args.keypoints),
        metrics_out=Path(args.metrics_out),
    )


if __name__ == "__main__":
    main()

