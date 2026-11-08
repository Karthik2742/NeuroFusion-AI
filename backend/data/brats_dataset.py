"""
NeuroFusion AI - BraTS 2020 PyTorch Dataset & DataLoader
Handles volumetric multi-sequence NIfTI scans for MRI Enhancement & Segmentation training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import cv2


class BraTSDataset(Dataset):
    """
    BraTS 2020 PyTorch Dataset.
    
    Supports:
        - Multi-modality reading: FLAIR, T1, T1CE, T2
        - Segmentation mask loading: Background (0), NCR/NET (1), ED (2), ET (4)
        - On-the-fly 2D slice extraction from 3D NIfTI volumes
        - Paired low-quality/high-quality degradation generation for Enhancement models
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        mode: str = "segmentation",  # "segmentation" or "enhancement"
        modality: str = "flair",    # "flair", "t1", "t1ce", "t2", or "all"
        slice_plane: str = "axial", # "axial", "coronal", "sagittal"
        slice_step: int = 2,        # Step size to sample slices (skip empty border slices)
        img_size: Tuple[int, int] = (224, 224),
        noise_std: float = 0.05,    # Noise level for synthetic enhancement degradation
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.mode = mode
        self.modality = modality.lower()
        self.slice_plane = slice_plane.lower()
        self.slice_step = slice_step
        self.img_size = img_size
        self.noise_std = noise_std
        self.transform = transform

        self.samples: List[dict[str, Any]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Scan dataset directory and build slice index."""
        if not self.dataset_dir.exists():
            logger.warning(f"Dataset path does not exist: {self.dataset_dir}")
            return

        # Search recursively for BraTS patient directories containing NIfTI files
        patient_dirs = []
        for root, dirs, files in os.walk(self.dataset_dir):
            if any(f.endswith(".nii") or f.endswith(".nii.gz") for f in files):
                # Verify if this directory has valid scan files
                flair_files = [f for f in files if "flair" in f.lower()]
                if flair_files:
                    patient_dirs.append(Path(root))

        patient_dirs = sorted(list(set(patient_dirs)))
        logger.info(f"Indexing BraTS dataset from {self.dataset_dir}: found {len(patient_dirs)} subject directories")

        for p_dir in patient_dirs:
            # Find modalities
            files = list(p_dir.glob("*.nii*"))
            file_map = {}
            for f in files:
                fname = f.name.lower()
                if "flair" in fname:
                    file_map["flair"] = f
                elif "t1ce" in fname or "t1_ce" in fname:
                    file_map["t1ce"] = f
                elif "t1" in fname:
                    file_map["t1"] = f
                elif "t2" in fname:
                    file_map["t2"] = f
                elif "seg" in fname:
                    file_map["seg"] = f

            # Verify minimum files needed
            if "flair" not in file_map:
                continue

            # Load FLAIR header to get depth
            try:
                flair_img = nib.load(str(file_map["flair"]))
                shape = flair_img.shape
                
                # Determine number of slices based on plane
                if self.slice_plane == "axial":
                    num_slices = shape[2]
                elif self.slice_plane == "coronal":
                    num_slices = shape[1]
                else:
                    num_slices = shape[0]

                # Filter out pure background slices (middle 70% of scan usually has brain)
                min_s = int(num_slices * 0.15)
                max_s = int(num_slices * 0.85)

                for slice_idx in range(min_s, max_s, self.slice_step):
                    self.samples.append({
                        "patient_id": p_dir.name,
                        "files": file_map,
                        "slice_idx": slice_idx,
                    })
            except Exception as e:
                logger.error(f"Failed to inspect {p_dir}: {e}")

        logger.info(f"BraTS Dataset indexed: total {len(self.samples)} slice samples available.")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalize foreground non-zero intensity range to [0, 1]."""
        fg = arr > 0
        if not np.any(fg):
            return np.zeros_like(arr, dtype=np.float32)
        v_min, v_max = arr[fg].min(), arr[fg].max()
        if v_max - v_min < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        norm = np.zeros_like(arr, dtype=np.float32)
        norm[fg] = (arr[fg] - v_min) / (v_max - v_min)
        return norm

    def _extract_slice(self, volume: np.ndarray, slice_idx: int) -> np.ndarray:
        """Extract 2D slice along chosen plane."""
        if self.slice_plane == "axial":
            return volume[:, :, slice_idx]
        elif self.slice_plane == "coronal":
            return volume[:, slice_idx, :]
        else:
            return volume[slice_idx, :, :]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample_info = self.samples[index]
        file_map = sample_info["files"]
        slice_idx = sample_info["slice_idx"]

        # Select target modality file
        mod_key = "flair" if self.modality not in file_map else self.modality
        mod_path = file_map[mod_key]

        # Load volume slice
        vol = nib.load(str(mod_path)).get_fdata()
        img_slice = self._extract_slice(vol, slice_idx)
        img_slice = cv2.resize(img_slice, self.img_size, interpolation=cv2.INTER_LINEAR)
        img_norm = self._normalize(img_slice)

        if self.mode == "segmentation":
            # Load segmentation ground truth mask if available
            mask_out = np.zeros(self.img_size, dtype=np.int64)
            if "seg" in file_map:
                seg_vol = nib.load(str(file_map["seg"])).get_fdata()
                seg_slice = self._extract_slice(seg_vol, slice_idx)
                seg_slice = cv2.resize(seg_slice, self.img_size, interpolation=cv2.INTER_NEAREST)

                # Map BraTS labels: 0=BG, 1=NCR/NET, 2=ED, 4=ET -> 0, 1, 2, 3
                mask_out[seg_slice == 1] = 1
                mask_out[seg_slice == 2] = 2
                mask_out[seg_slice == 4] = 3

            input_tensor = torch.from_numpy(img_norm).unsqueeze(0).float() # (1, H, W)
            target_tensor = torch.from_numpy(mask_out).long()              # (H, W)

            return {
                "input": input_tensor,
                "target": target_tensor,
                "patient_id": sample_info["patient_id"],
                "slice_idx": slice_idx,
            }

        else:
            # Mode: "enhancement" (Paired Low Quality -> High Quality target)
            target_norm = img_norm.copy()

            # Generate synthetic degraded input (downsample + add Rician/Gaussian noise)
            h, w = self.img_size
            downscaled = cv2.resize(img_norm, (h // 2, w // 2), interpolation=cv2.INTER_AREA)
            upscaled = cv2.resize(downscaled, (h, w), interpolation=cv2.INTER_CUBIC)

            # Add Rician noise
            noise1 = np.random.normal(0, self.noise_std, size=(h, w))
            noise2 = np.random.normal(0, self.noise_std, size=(h, w))
            degraded = np.sqrt((upscaled + noise1) ** 2 + noise2 ** 2)
            degraded = np.clip(degraded, 0, 1)

            input_tensor = torch.from_numpy(degraded).unsqueeze(0).float() # (1, H, W)
            target_tensor = torch.from_numpy(target_norm).unsqueeze(0).float() # (1, H, W)

            return {
                "input": input_tensor,
                "target": target_tensor,
                "patient_id": sample_info["patient_id"],
                "slice_idx": slice_idx,
            }


def get_brats_dataloader(
    dataset_dir: str | Path,
    mode: str = "segmentation",
    modality: str = "flair",
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    img_size: Tuple[int, int] = (224, 224),
) -> DataLoader:
    """Helper to instantiate a BraTS DataLoader."""
    dataset = BraTSDataset(
        dataset_dir=dataset_dir,
        mode=mode,
        modality=modality,
        img_size=img_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
