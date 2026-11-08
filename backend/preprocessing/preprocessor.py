"""
NeuroFusion AI - Module 2: MRI Preprocessing Pipeline
Comprehensive preprocessing for Brain MRI volumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy import ndimage

from backend.configs import config
from backend.utils.io import NIfTILoader


class MRIPreprocessor:
    """
    Production-grade MRI preprocessing pipeline.
    
    Supports:
        - N4 Bias Field Correction
        - Resampling to target spacing
        - Normalization (z-score, min-max, percentile)
        - Skull stripping
        - Cropping & Padding
        - Artifact removal
    """

    def __init__(self):
        self.loader = NIfTILoader()

    def run_pipeline(
        self,
        filepath: str | Path,
        steps: Optional[list[str]] = None,
        output_dir: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        Run the complete preprocessing pipeline on a NIfTI file.
        
        Args:
            filepath: Path to the NIfTI file
            steps: List of preprocessing steps to apply.
                   Options: 'bias_correction', 'resample', 'normalize',
                            'skull_strip', 'crop', 'pad'
                   If None, all steps are applied.
            output_dir: Directory to save preprocessed file
            
        Returns:
            Dictionary with preprocessed data, metadata, and step results
        """
        filepath = Path(filepath)
        logger.info(f"Starting preprocessing pipeline: {filepath.name}")

        if steps is None:
            steps = [
                "bias_correction",
                "normalize",
                "crop",
            ]

        # Load the volume
        data, header, affine = self.loader.load_nibabel(filepath)
        sitk_image = self.loader.load_sitk(filepath)

        results = {
            "input_file": str(filepath),
            "original_shape": list(data.shape),
            "steps_applied": [],
            "step_details": {},
        }

        # Apply each step
        for step in steps:
            logger.info(f"Applying: {step}")
            try:
                if step == "bias_correction":
                    sitk_image, step_info = self.bias_field_correction(sitk_image)
                    data = sitk.GetArrayFromImage(sitk_image).transpose()

                elif step == "resample":
                    sitk_image, step_info = self.resample(sitk_image)
                    data = sitk.GetArrayFromImage(sitk_image).transpose()

                elif step == "normalize":
                    data, step_info = self.normalize(data)

                elif step == "skull_strip":
                    data, step_info = self.skull_strip(data)

                elif step == "crop":
                    data, affine, step_info = self.crop_to_foreground(data, affine)

                elif step == "pad":
                    data, step_info = self.pad_volume(data)

                else:
                    logger.warning(f"Unknown step: {step}")
                    continue

                results["steps_applied"].append(step)
                results["step_details"][step] = step_info

            except Exception as e:
                logger.error(f"Step '{step}' failed: {e}")
                results["step_details"][step] = {"error": str(e)}

        results["output_shape"] = list(data.shape)

        # Save if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"preprocessed_{filepath.name}"

            if not output_path.name.endswith(".gz"):
                output_path = output_path.with_suffix(output_path.suffix + ".gz")

            self.loader.save_nifti(data, affine, output_path, header)
            results["output_file"] = str(output_path)

        logger.info(
            f"Preprocessing complete: {len(results['steps_applied'])} steps applied, "
            f"shape {results['original_shape']} → {results['output_shape']}"
        )

        return {**results, "data": data, "affine": affine, "header": header}

    def bias_field_correction(
        self,
        image: sitk.Image,
        shrink_factor: int = 3,
        max_iterations: list[int] | None = None,
    ) -> tuple[sitk.Image, dict]:
        """
        Apply N4 Bias Field Correction using SimpleITK.
        
        Corrects the low-frequency intensity non-uniformity (bias field)
        present in MRI images due to inhomogeneities in the magnetic field.
        
        Args:
            image: SimpleITK image
            shrink_factor: Factor to shrink image for faster computation
            max_iterations: Convergence iterations per level
            
        Returns:
            Corrected image and step information
        """
        if max_iterations is None:
            max_iterations = config.get(
                "preprocessing.bias_correction.max_iterations", [50, 50, 30, 20]
            )

        logger.info(f"N4 Bias Correction: shrink={shrink_factor}, iterations={max_iterations}")

        # Cast to float32 for N4
        input_image = sitk.Cast(image, sitk.sitkFloat32)

        # Create a mask of non-zero voxels
        mask = sitk.OtsuThreshold(input_image, 0, 1, 200)

        # Shrink for faster computation
        shrunk_image = sitk.Shrink(
            input_image, [shrink_factor] * input_image.GetDimension()
        )
        shrunk_mask = sitk.Shrink(
            mask, [shrink_factor] * mask.GetDimension()
        )

        # Apply N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(max_iterations)
        corrector.SetConvergenceThreshold(
            config.get("preprocessing.bias_correction.convergence_threshold", 0.001)
        )

        corrected = corrector.Execute(shrunk_image, shrunk_mask)

        # Get the bias field and apply to full-resolution image
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
        corrected_full = input_image / sitk.Exp(log_bias_field)

        step_info = {
            "method": "N4BiasFieldCorrection",
            "shrink_factor": shrink_factor,
            "max_iterations": max_iterations,
        }

        logger.info("N4 Bias Correction complete")
        return corrected_full, step_info

    def resample(
        self,
        image: sitk.Image,
        target_spacing: Optional[list[float]] = None,
        interpolation: str = "linear",
    ) -> tuple[sitk.Image, dict]:
        """
        Resample volume to target voxel spacing.
        
        Args:
            image: SimpleITK image
            target_spacing: Target spacing in mm [x, y, z]
            interpolation: 'linear', 'nearest', or 'bspline'
            
        Returns:
            Resampled image and step info
        """
        if target_spacing is None:
            target_spacing = config.get(
                "preprocessing.resampling.target_spacing", [1.0, 1.0, 1.0]
            )

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Compute new size
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]

        # Select interpolation method
        interp_map = {
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
            "bspline": sitk.sitkBSpline,
        }
        interp_method = interp_map.get(interpolation, sitk.sitkLinear)

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(interp_method)

        resampled = resampler.Execute(image)

        step_info = {
            "method": "Resampling",
            "original_spacing": list(original_spacing),
            "target_spacing": target_spacing,
            "original_size": list(original_size),
            "new_size": new_size,
            "interpolation": interpolation,
        }

        logger.info(
            f"Resampled: {list(original_spacing)} → {target_spacing}, "
            f"size {list(original_size)} → {new_size}"
        )
        return resampled, step_info

    def normalize(
        self,
        data: np.ndarray,
        method: str = "zscore",
        clip_range: Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Normalize intensity values.
        
        Args:
            data: Volume data
            method: 'zscore', 'minmax', or 'percentile'
            clip_range: Optional (lower, upper) percentile for clipping
            
        Returns:
            Normalized data and step info
        """
        if method is None:
            method = config.get("preprocessing.normalization.method", "zscore")

        original_range = (float(data.min()), float(data.max()))
        mask = data > 0  # Foreground mask

        if method == "zscore":
            # Z-score normalization on foreground only
            fg = data[mask]
            mean_val = fg.mean()
            std_val = fg.std()
            data = np.where(mask, (data - mean_val) / (std_val + 1e-8), 0)

        elif method == "minmax":
            fg = data[mask]
            min_val = fg.min()
            max_val = fg.max()
            data = np.where(
                mask,
                (data - min_val) / (max_val - min_val + 1e-8),
                0,
            )

        elif method == "percentile":
            p_low = config.get("preprocessing.normalization.percentile_lower", 0.5)
            p_high = config.get("preprocessing.normalization.percentile_upper", 99.5)
            fg = data[mask]
            low = np.percentile(fg, p_low)
            high = np.percentile(fg, p_high)
            data = np.clip(data, low, high)
            data = np.where(mask, (data - low) / (high - low + 1e-8), 0)

        if clip_range:
            data = np.clip(data, clip_range[0], clip_range[1])

        step_info = {
            "method": method,
            "original_range": original_range,
            "output_range": (float(data.min()), float(data.max())),
        }

        logger.info(f"Normalization ({method}): {original_range} → {step_info['output_range']}")
        return data.astype(np.float32), step_info

    def skull_strip(
        self,
        data: np.ndarray,
        method: str = "threshold",
    ) -> tuple[np.ndarray, dict]:
        """
        Remove skull/non-brain tissue from brain MRI.
        
        Args:
            data: Volume data
            method: 'threshold' (simple thresholding)
            
        Returns:
            Skull-stripped data and step info
        """
        original_nonzero = int(np.count_nonzero(data))

        if method == "threshold":
            # Simple threshold-based skull stripping
            # Works well for pre-processed data like BraTS
            threshold = np.percentile(data[data > 0], 2)
            mask = data > threshold

            # Morphological operations to clean up
            mask = ndimage.binary_fill_holes(mask)
            mask = ndimage.binary_erosion(mask, iterations=1)
            mask = ndimage.binary_dilation(mask, iterations=1)

            # Find largest connected component (the brain)
            labeled, num_features = ndimage.label(mask)
            if num_features > 1:
                sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
                largest = np.argmax(sizes) + 1
                mask = labeled == largest

            data = data * mask

        stripped_nonzero = int(np.count_nonzero(data))
        removed_pct = round(
            (1 - stripped_nonzero / max(original_nonzero, 1)) * 100, 1
        )

        step_info = {
            "method": method,
            "voxels_before": original_nonzero,
            "voxels_after": stripped_nonzero,
            "removed_percent": removed_pct,
        }

        logger.info(f"Skull strip: removed {removed_pct}% of non-zero voxels")
        return data.astype(np.float32), step_info

    def crop_to_foreground(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        margin: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Crop volume to the bounding box of the foreground region.
        
        Args:
            data: Volume data
            affine: Affine transformation matrix
            margin: Margin (voxels) around the bounding box
            
        Returns:
            Cropped data, updated affine, and step info
        """
        original_shape = data.shape

        # Find bounding box of non-zero region
        non_zero = np.argwhere(data > 0)
        if len(non_zero) == 0:
            return data, affine, {"method": "crop", "cropped": False}

        bbox_min = np.maximum(non_zero.min(axis=0) - margin, 0)
        bbox_max = np.minimum(
            non_zero.max(axis=0) + margin + 1,
            np.array(data.shape[:3])
        )

        # Crop
        slices = tuple(slice(int(mn), int(mx)) for mn, mx in zip(bbox_min, bbox_max))
        cropped = data[slices]

        # Update affine to reflect new origin
        new_affine = affine.copy()
        offset = affine[:3, :3] @ bbox_min
        new_affine[:3, 3] = affine[:3, 3] + offset

        step_info = {
            "method": "crop_to_foreground",
            "original_shape": list(original_shape),
            "cropped_shape": list(cropped.shape),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "margin": margin,
        }

        logger.info(f"Cropped: {list(original_shape)} → {list(cropped.shape)}")
        return cropped, new_affine, step_info

    def pad_volume(
        self,
        data: np.ndarray,
        target_shape: Optional[tuple[int, ...]] = None,
        mode: str = "constant",
    ) -> tuple[np.ndarray, dict]:
        """
        Pad volume to target shape.
        
        Args:
            data: Volume data
            target_shape: Desired shape. If None, pads to next power of 2.
            mode: Padding mode ('constant', 'edge', 'reflect')
            
        Returns:
            Padded data and step info
        """
        original_shape = data.shape

        if target_shape is None:
            # Pad to make dimensions divisible by 16 (for most networks)
            target_shape = tuple(
                int(np.ceil(s / 16) * 16) for s in data.shape[:3]
            )

        # Calculate padding
        pad_widths = []
        for orig, target in zip(data.shape[:3], target_shape[:3]):
            total_pad = max(target - orig, 0)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_widths.append((pad_before, pad_after))

        # Handle extra dimensions
        while len(pad_widths) < len(data.shape):
            pad_widths.append((0, 0))

        padded = np.pad(data, pad_widths, mode=mode)

        step_info = {
            "method": "pad",
            "original_shape": list(original_shape),
            "padded_shape": list(padded.shape),
            "target_shape": list(target_shape),
            "pad_widths": pad_widths,
            "mode": mode,
        }

        logger.info(f"Padded: {list(original_shape)} → {list(padded.shape)}")
        return padded, step_info
