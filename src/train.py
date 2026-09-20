"""Fold-aware training entry point for the RSNA knee classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any



def _require_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError("Training requires pandas, torch, and torchvision.") from error
    try:
        import torch.nn.functional as functional
    except ImportError as error:
        raise ImportError("Training requires torch.nn.functional.") from error
    return pd, torch, DataLoader, functional


def split_by_fold(train: Any, folds: Any, validation_fold: int) -> tuple[Any, Any]:
    """Join the persisted fold mapping and return train and validation rows."""
    merged = train.merge(folds, on="StudyInstanceUID", how="inner", validate="one_to_one")
    if merged["fold"].isna().any():
        raise ValueError("Every study must have a persisted fold assignment.")
    validation = merged[merged["fold"] == validation_fold].reset_index(drop=True)
    training = merged[merged["fold"] != validation_fold].reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError(f"Fold {validation_fold} produced an empty split.")
    return training, validation


def masked_bce_loss(logits: Any, targets: Any, mask: Any, functional: Any) -> Any:
    """Calculate BCE only where a label is present in train.csv."""
    element_loss = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    observed = mask.to(dtype=element_loss.dtype)
    return (element_loss * observed).sum() / observed.sum().clamp_min(1.0)


def run_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    torch: Any,
    functional: Any,
    training: bool,
) -> float:
    model.train(training)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = torch.as_tensor(batch["target"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.bool, device=device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = masked_bce_loss(logits, targets, mask, functional)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


def train_one_fold(args: argparse.Namespace) -> None:
    pandas, torch, data_loader, functional = _require_training_dependencies()
    try:
        from dataset import KneeStudyDataset, load_metadata
        from model import build_model
    except ImportError:
        from .dataset import KneeStudyDataset, load_metadata
        from .model import build_model

    data_dir = Path(args.data_dir)
    train, train_series = load_metadata(data_dir)
    folds = pandas.read_csv(data_dir / "folds.csv")
    training_rows, validation_rows = split_by_fold(train, folds, args.validation_fold)
    if set(training_rows["StudyInstanceUID"]) & set(validation_rows["StudyInstanceUID"]):
        raise AssertionError("Study leakage detected between training and validation folds.")

    training_dataset = KneeStudyDataset(
        training_rows,
        train_series,
        data_dir,
        image_size=args.image_size,
        anatomical_plane=args.anatomical_plane,
    )
    validation_dataset = KneeStudyDataset(
        validation_rows,
        train_series,
        data_dir,
        image_size=args.image_size,
        anatomical_plane=args.anatomical_plane,
    )
    training_loader = data_loader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = data_loader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.backbone, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        training_loss = run_epoch(model, training_loader, optimizer, device, torch, functional, training=True)
        validation_loss = run_epoch(model, validation_loader, optimizer, device, torch, functional, training=False)
        print(
            f"fold={args.validation_fold} epoch={epoch + 1}/{args.epochs} "
            f"train_loss={training_loss:.5f} validation_loss={validation_loss:.5f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": args.backbone,
                    "validation_fold": args.validation_fold,
                    "validation_loss": validation_loss,
                },
                output_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--anatomical-plane", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/best_fold.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train_one_fold(parse_args())
