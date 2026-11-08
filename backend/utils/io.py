"""
NeuroFusion AI - NIfTI I/O Utilities
Handles loading, saving, and validation of NIfTI medical imaging files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from loguru import logger


class NIfTILoader:
    """Unified NIfTI file loader supporting .nii and .nii.gz formats."""

    SUPPORTED_EXTENSIONS = {".nii", ".nii.gz"}

    @staticmethod
    def validate_file(filepath: str | Path) -> bool:
        """
        Validate that a file is a supported NIfTI format.
        
        Args:
            filepath: Path to the file to validate
            
        Returns:
            True if valid NIfTI file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return False

        suffix = "".join(filepath.suffixes).lower()
        if suffix not in NIfTILoader.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported format: {suffix}. Expected .nii or .nii.gz")
            return False

        try:
            img = nib.load(str(filepath))
            _ = img.header
            return True
        except Exception as e:
            logger.error(f"Invalid NIfTI file {filepath}: {e}")
            return False

    @staticmethod
    def load_nibabel(filepath: str | Path) -> tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
        """
        Load a NIfTI file using nibabel.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            Tuple of (volume_data, header, affine_matrix)
        """
        filepath = Path(filepath)
        logger.info(f"Loading NIfTI file: {filepath.name}")

        img = nib.load(str(filepath))
        data = img.get_fdata(dtype=np.float32)
        header = img.header
        affine = img.affine

        logger.info(
            f"Loaded volume: shape={data.shape}, "
            f"dtype={data.dtype}, "
            f"voxel_size={header.get_zooms()}"
        )
        return data, header, affine

    @staticmethod
    def load_sitk(filepath: str | Path) -> sitk.Image:
        """
        Load a NIfTI file using SimpleITK.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            SimpleITK Image object
        """
        filepath = Path(filepath)
        logger.info(f"Loading NIfTI (SimpleITK): {filepath.name}")

        image = sitk.ReadImage(str(filepath))
        logger.info(
            f"Loaded: size={image.GetSize()}, "
            f"spacing={image.GetSpacing()}, "
            f"origin={image.GetOrigin()}"
        )
        return image

    @staticmethod
    def save_nifti(
        data: np.ndarray,
        affine: np.ndarray,
        filepath: str | Path,
        header: Optional[nib.Nifti1Header] = None,
    ) -> Path:
        """
        Save a numpy array as a NIfTI file.
        
        Args:
            data: Volume data as numpy array
            affine: 4x4 affine transformation matrix
            filepath: Output file path
            header: Optional NIfTI header to preserve metadata
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if header is not None:
            img = nib.Nifti1Image(data, affine, header)
        else:
            img = nib.Nifti1Image(data, affine)

        nib.save(img, str(filepath))
        logger.info(f"Saved NIfTI: {filepath.name} (shape={data.shape})")
        return filepath

    @staticmethod
    def save_sitk(
        image: sitk.Image,
        filepath: str | Path,
    ) -> Path:
        """
        Save a SimpleITK image as NIfTI.
        
        Args:
            image: SimpleITK Image object
            filepath: Output file path
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(image, str(filepath))
        logger.info(f"Saved NIfTI (SimpleITK): {filepath.name}")
        return filepath

    @staticmethod
    def get_metadata(filepath: str | Path) -> dict:
        """
        Extract comprehensive metadata from a NIfTI file.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            Dictionary with all metadata
        """
        filepath = Path(filepath)
        img = nib.load(str(filepath))
        header = img.header
        data = img.get_fdata(dtype=np.float32)

        metadata = {
            "filename": filepath.name,
            "filepath": str(filepath),
            "shape": list(data.shape),
            "ndim": data.ndim,
            "dtype": str(data.dtype),
            "voxel_size": list(header.get_zooms()),
            "affine": img.affine.tolist(),
            "data_range": {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
            },
            "file_size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
            "orientation": "".join(nib.aff2axcodes(img.affine)),
            "is_3d": data.ndim == 3,
            "is_4d": data.ndim == 4,
            "num_voxels": int(np.prod(data.shape[:3])),
            "non_zero_voxels": int(np.count_nonzero(data)),
            "non_zero_ratio": float(np.count_nonzero(data) / np.prod(data.shape[:3])),
        }

        # Add spacing info from SimpleITK for more precision
        try:
            sitk_img = sitk.ReadImage(str(filepath))
            metadata["spacing"] = list(sitk_img.GetSpacing())
            metadata["origin"] = list(sitk_img.GetOrigin())
            metadata["direction"] = list(sitk_img.GetDirection())
        except Exception:
            metadata["spacing"] = list(header.get_zooms())

        return metadata


class BraTSDatasetLoader:
    """
    Specialized loader for the BraTS 2020 dataset.
    
    Expected structure:
        BraTS20_Training_XXX/
            BraTS20_Training_XXX_t1.nii.gz
            BraTS20_Training_XXX_t1ce.nii.gz
            BraTS20_Training_XXX_t2.nii.gz
            BraTS20_Training_XXX_flair.nii.gz
            BraTS20_Training_XXX_seg.nii.gz
    """

    MODALITIES = ["t1", "t1ce", "t2", "flair"]
    MODALITY_LABELS = {
        "t1": "T1-weighted",
        "t1ce": "T1-weighted Contrast Enhanced",
        "t2": "T2-weighted",
        "flair": "T2-FLAIR",
    }
    SEG_LABELS = {
        0: "Background",
        1: "NCR/NET (Necrotic/Non-Enhancing Tumor Core)",
        2: "ED (Peritumoral Edema)",
        4: "ET (GD-Enhancing Tumor)",
    }

    def __init__(self, root_dir: str | Path):
        """
        Initialize BraTS dataset loader.
        
        Args:
            root_dir: Root directory containing BraTS patient folders
        """
        self.root_dir = Path(root_dir)
        self.patients: list[dict] = []
        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """Scan the dataset directory and index all patients."""
        if not self.root_dir.exists():
            logger.warning(f"BraTS directory not found: {self.root_dir}")
            return

        patient_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith("BraTS")
        ])

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            patient_data = {
                "id": patient_id,
                "dir": str(patient_dir),
                "modalities": {},
                "has_seg": False,
            }

            for modality in self.MODALITIES:
                mod_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
                if mod_file.exists():
                    patient_data["modalities"][modality] = str(mod_file)

            seg_file = patient_dir / f"{patient_id}_seg.nii.gz"
            if seg_file.exists():
                patient_data["has_seg"] = True
                patient_data["seg_path"] = str(seg_file)

            if patient_data["modalities"]:
                self.patients.append(patient_data)

        logger.info(
            f"BraTS dataset indexed: {len(self.patients)} patients "
            f"in {self.root_dir}"
        )

    def load_patient(
        self, patient_id: str
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Load all modalities for a single patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'BraTS20_Training_001')
            
        Returns:
            Dictionary mapping modality name to (data, affine) tuple
        """
        patient = next(
            (p for p in self.patients if p["id"] == patient_id), None
        )
        if patient is None:
            raise ValueError(f"Patient {patient_id} not found in dataset")

        result = {}
        for modality, filepath in patient["modalities"].items():
            data, _, affine = NIfTILoader.load_nibabel(filepath)
            result[modality] = (data, affine)

        if patient.get("has_seg"):
            seg_data, _, seg_affine = NIfTILoader.load_nibabel(
                patient["seg_path"]
            )
            result["seg"] = (seg_data, seg_affine)

        logger.info(
            f"Loaded patient {patient_id}: "
            f"{list(result.keys())} modalities"
        )
        return result

    def get_patient_list(self) -> list[str]:
        """Get list of all patient IDs."""
        return [p["id"] for p in self.patients]

    def get_dataset_stats(self) -> dict:
        """Get high-level dataset statistics."""
        total_patients = len(self.patients)
        patients_with_seg = sum(1 for p in self.patients if p["has_seg"])
        modality_counts = {mod: 0 for mod in self.MODALITIES}

        for patient in self.patients:
            for mod in patient["modalities"]:
                modality_counts[mod] += 1

        return {
            "total_patients": total_patients,
            "patients_with_segmentation": patients_with_seg,
            "modality_counts": modality_counts,
            "root_directory": str(self.root_dir),
        }
