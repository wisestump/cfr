#!/usr/bin/env python3
"""Train MegaDescriptor-L-384 on a mix of full and foreground-only images.

This is the public wildlife-tools recipe
(baselines/training/MegaDescriptor-L-384.ipynb) with one change: each training
image is swapped for its foreground-only version (from make_foregrounds.py)
with probability --foreground-prob. Images without a foreground version always
use the full image. --foreground-prob 0 gives the full-image baseline.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from itertools import chain
from pathlib import Path

import pandas as pd
import timm
import torch
import torchvision.transforms as T
from torch.optim import SGD
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed

from common import TRAIN_DATASETS, foreground_path


class MixedForegroundDataset(WildlifeDataset):
    """WildlifeDataset that returns the foreground image with a fixed probability."""

    def __init__(self, metadata, root, foreground_root, foreground_prob, transform):
        super().__init__(metadata=metadata, root=root, transform=transform)
        self.foreground_prob = foreground_prob
        self.foreground_paths = [None] * len(self.metadata)
        if foreground_prob > 0:
            for i, path in enumerate(self.metadata["path"]):
                candidate = foreground_path(foreground_root, path)
                if candidate.is_file():
                    self.foreground_paths[i] = str(candidate)

    @property
    def foreground_coverage(self) -> float:
        return sum(p is not None for p in self.foreground_paths) / len(self)

    def __getitem__(self, idx):
        foreground = self.foreground_paths[idx]
        if foreground is not None and random.random() < self.foreground_prob:
            img = self.get_image(foreground)
        else:
            img = self.get_image(os.path.join(self.root, self.metadata.iloc[idx]["path"]))
        return self.transform(img), self.labels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder that contains images/size-518 and metadata/combined/combined_all.csv.",
    )
    parser.add_argument("--foreground-root", type=Path, help="Default: <data-root>/images/size-518-fg.")
    parser.add_argument("--foreground-prob", type=float, required=True, help="E.g. 0, 0.25, 0.5 or 0.75.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Use only N training images (smoke test).")
    args = parser.parse_args()
    if not 0.0 <= args.foreground_prob <= 1.0:
        parser.error("--foreground-prob must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    foreground_root = (args.foreground_root or data_root / "images" / "size-518-fg").expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Dataset configuration (as in the public notebook, restricted to the 22 training datasets)
    metadata = pd.read_csv(data_root / "metadata" / "combined" / "combined_all.csv")
    metadata = metadata.query('split == "train"')
    metadata = metadata[metadata["dataset"].isin(TRAIN_DATASETS)]
    if args.max_samples:
        metadata = metadata.sample(n=args.max_samples, random_state=args.seed)
    transform = T.Compose(
        [
            T.Resize(size=(384, 384)),
            T.RandAugment(num_ops=2, magnitude=20),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = MixedForegroundDataset(
        metadata=metadata,
        root=str(data_root / "images" / "size-518"),
        foreground_root=foreground_root,
        foreground_prob=args.foreground_prob,
        transform=transform,
    )
    print(
        f"{len(dataset)} images, {dataset.num_classes} identities, "
        f"foreground available for {dataset.foreground_coverage:.1%}, foreground prob {args.foreground_prob}",
        flush=True,
    )

    # Backbone and loss configuration
    backbone = timm.create_model("swin_large_patch4_window12_384", num_classes=0, pretrained=True)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 384, 384)
        embedding_size = backbone(dummy_input).shape[1]
    objective = ArcFaceLoss(num_classes=dataset.num_classes, embedding_size=embedding_size, margin=0.5, scale=64)

    # Optimizer and scheduler configuration
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)
    min_lr = optimizer.defaults.get("lr") * 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=min_lr)

    log_path = output_dir / "train_log.csv"

    def on_epoch_end(trainer, epoch_data):
        trainer.save(str(output_dir), "checkpoint.pth")
        with log_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([trainer.epoch, epoch_data["train_loss_epoch_avg"], scheduler.get_last_lr()[0]])
        print(f"epoch {trainer.epoch}: loss {epoch_data['train_loss_epoch_avg']:.4f}", flush=True)

    # Setup training
    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_workers=args.num_workers,
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epoch_callback=on_epoch_end,
    )

    # Resume after a job time limit: BasicTrainer.train() runs `epochs` more epochs.
    checkpoint = output_dir / "checkpoint.pth"
    if checkpoint.is_file():
        trainer.load(str(checkpoint))
        trainer.epochs = args.epochs - trainer.epoch
        print(f"Resumed from {checkpoint} at epoch {trainer.epoch}", flush=True)
    else:
        with log_path.open("w", newline="") as stream:
            csv.writer(stream).writerow(["epoch", "train_loss", "next_epoch_lr"])

    trainer.train()

    # Backbone weights only; load into timm "swin_large_patch4_window12_384" (num_classes=0).
    torch.save(backbone.state_dict(), output_dir / "megadescriptor-l-384-backbone.pt")


if __name__ == "__main__":
    main()
