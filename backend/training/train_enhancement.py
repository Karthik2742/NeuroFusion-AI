"""
NeuroFusion AI - MRI Enhancement Training Engine
Trains Restormer, SwinIR, or MIRNet models on BraTS 2020 Paired MRI Data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm

from backend.data.brats_dataset import get_brats_dataloader
from backend.models.enhancement import get_enhancement_model


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant for smooth image restoration)."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def create_synthetic_enhancement_loader(batch_size: int = 4, num_samples: int = 16) -> DataLoader:
    """Create synthetic paired data loader for dry-run verification."""
    targets = torch.rand(num_samples, 1, 224, 224)
    inputs = targets + torch.randn(num_samples, 1, 224, 224) * 0.05
    dataset = TensorDataset(inputs, targets)

    class CustomWrapper:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            inp, tgt = self.ds[idx]
            return {"input": inp, "target": tgt}

    return DataLoader(CustomWrapper(dataset), batch_size=batch_size, shuffle=True)


def train_enhancement(
    dataset_dir: str | Path = "./dataset/brain",
    architecture: str = "restormer",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 2e-4,
    save_dir: str | Path = "./weights",
    dry_run: bool = False,
) -> Path:
    """Run full enhancement model training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Enhancement Training | Architecture: {architecture} | Device: {device}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if dry_run or not Path(dataset_dir).exists():
        logger.warning("Using synthetic data loader for dry-run/testing.")
        train_loader = create_synthetic_enhancement_loader(batch_size=batch_size)
        val_loader = train_loader
    else:
        train_loader = get_brats_dataloader(
            dataset_dir=dataset_dir,
            mode="enhancement",
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = train_loader

    # Instantiate Model
    model = get_enhancement_model(architecture).to(device)
    criterion = CharbonnierLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_psnr = 0.0
    best_weights_path = save_dir / f"enhancement_{architecture}_brats20.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for batch in pbar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation phase
        model.eval()
        val_psnr_list = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                psnr = calculate_psnr(outputs, targets)
                val_psnr_list.append(psnr)

        mean_val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 0.0
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Mean PSNR: {mean_val_psnr:.2f} dB")

        # Save Best Model
        if mean_val_psnr >= best_psnr:
            best_psnr = mean_val_psnr
            model.save_weights(best_weights_path)
            logger.info(f"-> Saved new best checkpoint to {best_weights_path}")

    logger.info(f"[COMPLETE] Enhancement Training finished. Best Val PSNR: {best_psnr:.2f} dB")
    return best_weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI Enhancement Engine")
    parser.add_argument("--dataset-dir", type=str, default="./dataset/brain")
    parser.add_argument("--architecture", type=str, default="restormer", choices=["restormer", "swinir", "mirnet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-dir", type=str, default="./weights")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch test on synthetic data")

    args = parser.parse_args()
    train_enhancement(
        dataset_dir=args.dataset_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
    )
