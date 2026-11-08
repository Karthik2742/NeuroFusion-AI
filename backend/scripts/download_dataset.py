"""
NeuroFusion AI - BraTS 2020 Dataset Ingestion & Downloader Script
Downloads 'awsaf49/brats20-dataset-training-validation' from Kaggle or verifies existing dataset files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import zipfile

from loguru import logger


DATASET_SLUG = "awsaf49/brats20-dataset-training-validation"
DEFAULT_TARGET_DIR = Path("./dataset/brain")


def verify_dataset(target_dir: Path) -> bool:
    """Verify if BraTS 2020 NIfTI files exist in the dataset directory."""
    if not target_dir.exists():
        return False

    nii_files = list(target_dir.rglob("*.nii")) + list(target_dir.rglob("*.nii.gz"))
    if len(nii_files) > 10:
        logger.info(f"[SUCCESS] Dataset verified at {target_dir}: found {len(nii_files)} NIfTI files.")
        return True
    return False


def download_brats(target_dir: Path = DEFAULT_TARGET_DIR) -> bool:
    """Attempt to download BraTS 2020 dataset from Kaggle."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if verify_dataset(target_dir):
        logger.info(f"Dataset already exists in {target_dir}. Skipping download.")
        return True

    logger.info(f"Attempting to download BraTS 2020 dataset ({DATASET_SLUG}) from Kaggle...")

    # Method 1: Try kagglehub
    try:
        import kagglehub
        logger.info("Using kagglehub to download dataset...")
        download_path = kagglehub.dataset_download(DATASET_SLUG)
        logger.info(f"Downloaded via kagglehub to {download_path}")

        # Move/copy downloaded files to target_dir
        for item in Path(download_path).glob("*"):
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        if verify_dataset(target_dir):
            return True
    except Exception as e:
        logger.warning(f"kagglehub download failed or not installed: {e}")

    # Method 2: Try kaggle CLI / API
    try:
        import kaggle
        logger.info("Using Kaggle API to download dataset...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_SLUG, path=str(target_dir), unzip=True)
        if verify_dataset(target_dir):
            return True
    except Exception as e:
        logger.warning(f"Kaggle API download failed: {e}")

    # Manual instructions if download API fails
    logger.error("=" * 70)
    logger.error("AUTOMATIC KAGGLE DOWNLOAD REQUIRES KAGGLE CREDENTIALS")
    logger.error("=" * 70)
    logger.error("To download automatically, set up Kaggle API key:")
    logger.error("  1. Create account on Kaggle.com -> Account -> 'Create New API Token' (kaggle.json)")
    logger.error(f"  2. Place kaggle.json in {Path.home() / '.kaggle/kaggle.json'}")
    logger.error("  3. Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
    logger.error("")
    logger.error("MANUAL DOWNLOAD ALTERNATIVE:")
    logger.error(f"  1. Go to: https://www.kaggle.com/datasets/{DATASET_SLUG}")
    logger.error(f"  2. Download & unzip into: {target_dir.resolve()}")
    logger.error("=" * 70)

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & Ingest BraTS 2020 Dataset from Kaggle")
    parser.add_argument("--target-dir", type=str, default="./dataset/brain", help="Directory to save dataset")
    args = parser.parse_args()

    success = download_brats(Path(args.target_dir))
    if not success:
        sys.exit(1)
