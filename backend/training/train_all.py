"""
NeuroFusion AI - Complete Training Launcher
Unified CLI interface to download/verify datasets and train Segmentation & Enhancement models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from loguru import logger

from backend.scripts.download_dataset import download_brats
from backend.training.train_segmentation import train_segmentation
from backend.training.train_enhancement import train_enhancement


def main():
    parser = argparse.ArgumentParser(description="NeuroFusion AI - Full Model Training Launcher")
    parser.add_argument("--dataset-dir", type=str, default="./dataset/brain", help="Dataset directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download attempt")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "segmentation", "enhancement"], help="Target task to train")
    parser.add_argument("--seg-arch", type=str, default="unet", choices=["unet", "attention_unet", "unet_plus"], help="Segmentation architecture")
    parser.add_argument("--enh-arch", type=str, default="restormer", choices=["restormer", "swinir", "mirnet"], help="Enhancement architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save-dir", type=str, default="./weights", help="Directory to save weights")
    parser.add_argument("--dry-run", action="store_true", help="Perform 1-epoch validation test with synthetic data")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("        NeuroFusion AI - Training Pipeline Launcher")
    logger.info("=" * 60)

    # 1. Dataset Verification / Ingestion
    if not args.dry_run and not args.skip_download:
        logger.info("[Step 1/3] Verifying / Downloading BraTS 2020 Dataset...")
        has_dataset = download_brats(Path(args.dataset_dir))
        if not has_dataset:
            logger.warning("Dataset verification incomplete. Proceeding with dry-run / fallback.")

    # 2. Segmentation Training
    if args.mode in ["all", "segmentation"]:
        logger.info(f"[Step 2/3] Launching Segmentation Training ({args.seg_arch})...")
        train_segmentation(
            dataset_dir=args.dataset_dir,
            architecture=args.seg_arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            dry_run=args.dry_run,
        )

    # 3. Enhancement Training
    if args.mode in ["all", "enhancement"]:
        logger.info(f"[Step 3/3] Launching Enhancement Training ({args.enh_arch})...")
        train_enhancement(
            dataset_dir=args.dataset_dir,
            architecture=args.enh_arch,
            epochs=args.epochs,
            batch_size=max(1, args.batch_size // 2),  # Smaller batch size for heavy transformers
            save_dir=args.save_dir,
            dry_run=args.dry_run,
        )

    logger.info("=" * 60)
    logger.info("        [COMPLETE] All Training Operations Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
