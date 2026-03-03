import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    DEVICE as CONFIG_DEVICE,
    KEYPOINT_DIR,
    LR,
    METADATA_PATH,
    NUM_EPOCHS,
    NUM_WORKERS,
    WEIGHT_DECAY,
)
from dataset import WLASLDataset, Standardize, collate_fn, compute_normalization_stats
from model import SignLanguageLSTM


def get_device() -> torch.device:
    if CONFIG_DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_datasets(
    metadata_path: Path,
    keypoint_dir: Path,
) -> Tuple[WLASLDataset, WLASLDataset, Dict[str, int]]:
    # First build train to get gloss mapping
    train_ds = WLASLDataset(
        metadata_path=metadata_path,
        keypoint_dir=keypoint_dir,
        split="train",
        gloss_to_idx=None,
        transform=None,
    )

    gloss_to_idx = train_ds.gloss_to_idx

    # Compute normalization stats on train
    mean, std = compute_normalization_stats(train_ds)
    transform = Standardize(mean, std)
    train_ds.transform = transform

    # Build validation dataset sharing same gloss mapping and transform
    val_ds = WLASLDataset(
        metadata_path=metadata_path,
        keypoint_dir=keypoint_dir,
        split="val",
        gloss_to_idx=gloss_to_idx,
        transform=transform,
    )

    return train_ds, val_ds, gloss_to_idx


def evaluate(
    model: SignLanguageLSTM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, lengths, labels in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits, _ = model(x, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    return avg_loss, acc


def train(
    metadata_path: Path,
    keypoint_dir: Path,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    num_workers: int = NUM_WORKERS,
    best_model_path: Path = BEST_MODEL_PATH,
) -> None:
    device = get_device()

    train_ds, val_ds, gloss_to_idx = build_datasets(
        metadata_path=metadata_path, keypoint_dir=keypoint_dir
    )

    num_classes = len(gloss_to_idx)
    model = SignLanguageLSTM(num_classes=num_classes)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, lengths, labels in loop:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(x, lengths)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            avg_train_loss = total_loss / max(total_examples, 1)
            train_acc = total_correct / max(total_examples, 1)
            loop.set_postfix(loss=avg_train_loss, acc=train_acc)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "gloss_to_idx": gloss_to_idx,
                },
                best_model_path,
            )
            print(f"Saved new best model to {best_model_path} (val_acc={val_acc:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM sign language classifier.")
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
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=LR, help="Learning rate."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay (L2 regularization).",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    keypoint_dir = Path(args.keypoints)
    best_model_path = BEST_MODEL_PATH

    train(
        metadata_path=metadata_path,
        keypoint_dir=keypoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=NUM_WORKERS,
        best_model_path=best_model_path,
    )


if __name__ == "__main__":
    main()

