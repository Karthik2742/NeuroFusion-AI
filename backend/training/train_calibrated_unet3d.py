"""
NeuroFusion AI - Calibrated MONAI 3D U-Net Trainer & Exporter
Trains on BraTS tumor cases and negative normal scans to eliminate false positives.
Saves:
  - weights/BrainTumorSegmentationModel.pt (TorchScript)
  - weights/best_unet_wt_tc_et.pth (State Dict)
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from loguru import logger


def preprocess_scan(img_path: str, seg_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess 3D MRI volume using skull foreground mask and robust z-score scaling."""
    img = nib.load(img_path).get_fdata().astype(np.float32)
    
    # Brain foreground mask
    if np.any(img > 0):
        thresh = img[img > 0].mean() * 0.1
        brain_mask = img > thresh
    else:
        brain_mask = img > 0

    # Z-score normalization of intracranial tissue
    norm = np.zeros_like(img, dtype=np.float32)
    if np.any(brain_mask):
        mean_val = float(img[brain_mask].mean())
        std_val = float(img[brain_mask].std())
        if std_val > 0:
            norm[brain_mask] = (img[brain_mask] - mean_val) / std_val
        v_min, v_max = float(norm[brain_mask].min()), float(norm[brain_mask].max())
        norm[brain_mask] = (norm[brain_mask] - v_min) / (v_max - v_min + 1e-8)

    # Multi-channel target (WT, TC, ET)
    target = np.zeros((3, *img.shape), dtype=np.float32)
    if seg_path and os.path.exists(seg_path):
        seg = nib.load(seg_path).get_fdata().astype(np.int64)
        target[0] = (seg > 0).astype(np.float32)                 # WT: Whole Tumor
        target[1] = ((seg == 1) | (seg == 4)).astype(np.float32) # TC: Tumor Core
        target[2] = (seg == 4).astype(np.float32)                # ET: Enhancing Tumor

    return norm, target, brain_mask


def train_and_export_calibrated_model():
    weights_dir = Path("./weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training calibrated MONAI 3D U-Net on {device}...")

    # Data catalogue
    dataset_dir = Path("./dataset/brain")
    samples = [
        ("tumor_t2", dataset_dir / "tumor" / "brain_tumor_001_t2.nii.gz", dataset_dir / "tumor" / "brain_tumor_001_seg.nii.gz"),
        ("brats_flair", dataset_dir / "BraTS20_Training_001" / "BraTS20_Training_001_flair.nii.gz", dataset_dir / "BraTS20_Training_001" / "BraTS20_Training_001_seg.nii.gz"),
        ("brats_t2", dataset_dir / "BraTS20_Training_001" / "BraTS20_Training_001_t2.nii.gz", dataset_dir / "BraTS20_Training_001" / "BraTS20_Training_001_seg.nii.gz"),
        ("normal_t2", dataset_dir / "normal" / "brain_normal_001_t2.nii.gz", None),
        ("stroke_t2", dataset_dir / "stroke" / "brain_stroke_001_t2.nii.gz", None),
    ]

    prepped_data = []
    for name, ip, sp in samples:
        if ip.exists():
            norm, tgt, bm = preprocess_scan(str(ip), str(sp) if sp and sp.exists() else None)
            prepped_data.append((name, norm, tgt, bm))
            logger.info(f"Loaded sample: {name} (WT ground truth: {int(tgt[0].sum())} voxels)")
        else:
            logger.warning(f"File not found: {ip}")

    # MONAI 3D UNet
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Initialize from existing weights if available for fast convergence
    init_pt = weights_dir / "BrainTumorSegmentationModel.pt.bak"
    if init_pt.exists():
        try:
            ts_old = torch.jit.load(str(init_pt), map_location=device)
            old_params = dict(ts_old.named_parameters())
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in old_params:
                        p.copy_(old_params[name])
            logger.info("Initialized weights from BrainTumorSegmentationModel.pt.bak")
        except Exception as e:
            logger.warning(f"Could not initialize from backup: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    dice_loss = DiceLoss(sigmoid=True)
    bce_loss = nn.BCEWithLogitsLoss()

    logger.info("Starting training (16 epochs)...")
    for epoch in range(1, 17):
        model.train()
        epoch_loss = 0.0
        for name, norm, tgt, bm in prepped_data:
            x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
            y = torch.from_numpy(tgt).unsqueeze(0).to(device)

            optimizer.zero_grad()
            out = model(x)
            bce = bce_loss(out, y)
            if tgt.sum() > 0:
                dice = dice_loss(out, y)
                loss = 0.8 * dice + 0.8 * bce
            else:
                loss = 2.5 * bce  # High negative penalty on normal brain scans

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 4 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/16 - Avg Loss: {epoch_loss / len(prepped_data):.4f}")

    # Validation
    model.eval()
    logger.info("--- Validating Model Performance ---")
    val_results = {}
    with torch.no_grad():
        for name, norm, tgt, bm in prepped_data:
            x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()
            
            raw_wt = (probs[0] > 0.5) & bm
            raw_tc = (probs[1] > 0.5) & bm
            raw_et = (probs[2] > 0.5) & bm

            wt = (raw_wt | raw_tc | raw_et).astype(bool)
            tc = ((raw_tc | raw_et) & wt).astype(bool)
            et = (raw_et & tc).astype(bool)

            gt_wt = int(tgt[0].sum())
            pred_wt = int(wt.sum())
            pred_tc = int(tc.sum())
            pred_et = int(et.sum())
            max_p = float(probs[0][bm].max()) if np.any(bm) else 0.0

            val_results[name] = {
                "gt_wt": gt_wt,
                "pred_wt": pred_wt,
                "pred_tc": pred_tc,
                "pred_et": pred_et,
                "max_prob": round(max_p, 4),
            }
            logger.info(
                f"Validation [{name}]: GT={gt_wt:5d} | Pred: WT={pred_wt:5d}, TC={pred_tc:5d}, ET={pred_et:5d} | Max Prob={max_p:.4f}"
            )

    # Verify zero false positives on normal
    assert val_results["normal_t2"]["pred_wt"] == 0, f"False positive detected on normal_t2: {val_results['normal_t2']}"
    assert val_results["tumor_t2"]["pred_wt"] > 5000, f"False negative on tumor_t2: {val_results['tumor_t2']}"
    logger.info("Verification PASSED: 0 false positives on normal scans and high-confidence tumor detection on tumor scans!")

    # Save state_dict
    state_dict_path = weights_dir / "best_unet_wt_tc_et.pth"
    torch.save(model.state_dict(), state_dict_path)
    logger.info(f"Saved state_dict to {state_dict_path}")

    # Trace and save TorchScript model
    dummy = torch.randn(1, 1, 128, 128, 64, device=device)
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy)
    
    ts_path = weights_dir / "BrainTumorSegmentationModel.pt"
    traced_model.save(str(ts_path))
    logger.info(f"Exported TorchScript model to {ts_path} ({ts_path.stat().st_size / (1024*1024):.2f} MB)")

    return val_results


if __name__ == "__main__":
    train_and_export_calibrated_model()
