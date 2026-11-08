"""
NeuroFusion AI - Train All Brain Models Suite
Trains all brain MRI ROI segmentation (UNet, Attention UNet, UNet++) and
enhancement transformer architectures (Restormer, SwinIR, MIRNet).
"""

from pathlib import Path
from loguru import logger
from backend.training.train_segmentation import train_segmentation
from backend.training.train_enhancement import train_enhancement

def train_all_brain():
    dataset_dir = Path("./dataset/brain")
    save_dir = Path("./weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    seg_architectures = ["unet", "attention_unet", "unet_plus"]
    enh_architectures = ["restormer", "swinir", "mirnet"]

    logger.info("============================================================")
    logger.info("      NeuroFusion AI - Training All Brain MRI Models        ")
    logger.info("============================================================")

    # 1. Train all segmentation models
    for arch in seg_architectures:
        logger.info(f"[Brain ROI Segmentation] Training architecture: {arch.upper()}")
        weights_path = train_segmentation(
            dataset_dir=dataset_dir,
            architecture=arch,
            epochs=1,
            batch_size=4,
            save_dir=save_dir,
            dry_run=not (dataset_dir / "BraTS20_Training_001").exists()
        )
        logger.info(f"-> Saved: {weights_path}")

    # 2. Train all enhancement models
    for arch in enh_architectures:
        logger.info(f"[Brain MRI Enhancement] Training architecture: {arch.upper()}")
        weights_path = train_enhancement(
            dataset_dir=dataset_dir,
            architecture=arch,
            epochs=1,
            batch_size=2,
            save_dir=save_dir,
            dry_run=not (dataset_dir / "BraTS20_Training_001").exists()
        )
        logger.info(f"-> Saved: {weights_path}")

    logger.info("============================================================")
    logger.info(" [SUCCESS] All Brain Models Trained & Saved to ./weights    ")
    logger.info("============================================================")

if __name__ == "__main__":
    train_all_brain()
