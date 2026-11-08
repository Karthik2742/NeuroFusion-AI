"""
NeuroFusion AI - ROI Segmentation Training Engine
Trains UNet, Attention UNet, or UNet++ on BraTS 2020 Volumetric Data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm

from backend.data.brats_dataset import get_brats_dataloader, BraTSDataset
from backend.models.segmentation.segmenter import get_segmentation_model


class CombinedDiceCELoss(nn.Module):
    """Combined Multi-class Dice Loss + Cross Entropy Loss."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cross Entropy
        ce = self.ce_loss(pred, target)

        # Softmax for Dice calculation
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Dice Loss per class
        dice_loss = 0.0
        for c in range(self.num_classes):
            p_c = pred_softmax[:, c]
            t_c = target_onehot[:, c]
            intersection = (p_c * t_c).sum()
            total = p_c.sum() + t_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (total + self.smooth)
            dice_loss += (1.0 - dice_c)

        dice_loss = dice_loss / self.num_classes
        return 0.5 * ce + 0.5 * dice_loss


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
    """Calculate Multi-class Dice Scores."""
    pred_class = torch.argmax(pred, dim=1)
    scores = {}

    for c in range(1, num_classes):
        p_mask = (pred_class == c)
        t_mask = (target == c)
        intersection = (p_mask & t_mask).sum().item()
        total = p_mask.sum().item() + t_mask.sum().item()
        score = (2.0 * intersection) / (total + 1e-8) if total > 0 else 1.0
        class_name = ["BG", "NCR_NET", "ED", "ET"][c] if c < 4 else f"Class_{c}"
        scores[class_name] = score

    scores["Mean_Dice"] = float(np.mean(list(scores.values())))
    return scores


def create_synthetic_seg_loader(batch_size: int = 4, num_samples: int = 16) -> DataLoader:
    """Create a lightweight synthetic dataset for dry-run verification."""
    inputs = torch.randn(num_samples, 1, 224, 224)
    targets = torch.randint(0, 4, (num_samples, 224, 224))
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


def train_segmentation(
    dataset_dir: str | Path = "./dataset/brain",
    architecture: str = "unet",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_dir: str | Path = "./weights",
    dry_run: bool = False,
) -> Path:
    """Run full segmentation training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Segmentation Training | Architecture: {architecture} | Device: {device}")

    # Load Data
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if dry_run or not Path(dataset_dir).exists():
        logger.warning("Using synthetic data loader for dry-run/testing.")
        train_loader = create_synthetic_seg_loader(batch_size=batch_size)
        val_loader = train_loader
    else:
        train_loader = get_brats_dataloader(
            dataset_dir=dataset_dir,
            mode="segmentation",
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = train_loader

    # Instantiate Model
    model = get_segmentation_model(architecture, in_channels=1, out_channels=4).to(device)
    criterion = CombinedDiceCELoss(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dice = 0.0
    best_weights_path = save_dir / f"segmentation_{architecture}_brats20.pth"

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
        val_dice_list = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                scores = calculate_dice_score(outputs, targets)
                val_dice_list.append(scores["Mean_Dice"])

        mean_val_dice = float(np.mean(val_dice_list)) if val_dice_list else 0.0
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Mean Dice: {mean_val_dice:.4f}")

        # Save Best Model
        if mean_val_dice >= best_dice:
            best_dice = mean_val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "epoch": epoch,
                "val_dice": mean_val_dice,
            }, str(best_weights_path))
            logger.info(f"-> Saved new best checkpoint to {best_weights_path}")

    logger.info(f"[COMPLETE] Segmentation Training finished. Best Val Dice: {best_dice:.4f}")
    return best_weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI ROI Segmentation Engine")
    parser.add_argument("--dataset-dir", type=str, default="./dataset/brain")
    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "unet_plus", "attention_unet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="./weights")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch test on synthetic data")

    args = parser.parse_args()
    train_segmentation(
        dataset_dir=args.dataset_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
    )
