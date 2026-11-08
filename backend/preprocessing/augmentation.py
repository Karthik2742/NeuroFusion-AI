"""
NeuroFusion AI - Data Augmentation Pipeline
MRI-specific augmentation transforms for training data expansion.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger
from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter

from backend.configs import config


class MRIAugmentor:
    """
    MRI-specific data augmentation pipeline.
    
    All transforms are designed to produce anatomically plausible results
    and operate on 3D volumes.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed or config.get("gpu.seed", 42))

    def augment(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        transforms: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply a sequence of augmentation transforms.
        
        Args:
            data: 3D volume (H, W, D)
            mask: Optional segmentation mask (same shape)
            transforms: List of transforms to apply. If None, applies all.
            
        Returns:
            Augmented (data, mask) pair
        """
        if transforms is None:
            transforms = [
                "rotation", "flip", "brightness",
                "contrast", "gamma", "scaling",
            ]

        aug_data = data.copy()
        aug_mask = mask.copy() if mask is not None else None

        for t in transforms:
            if self.rng.random() > 0.5:  # 50% chance per transform
                try:
                    if t == "rotation":
                        aug_data, aug_mask = self.random_rotation(aug_data, aug_mask)
                    elif t == "flip":
                        aug_data, aug_mask = self.random_flip(aug_data, aug_mask)
                    elif t == "brightness":
                        aug_data = self.random_brightness(aug_data)
                    elif t == "contrast":
                        aug_data = self.random_contrast(aug_data)
                    elif t == "gamma":
                        aug_data = self.random_gamma(aug_data)
                    elif t == "scaling":
                        aug_data, aug_mask = self.random_scaling(aug_data, aug_mask)
                    elif t == "elastic":
                        aug_data, aug_mask = self.elastic_deformation(aug_data, aug_mask)
                    elif t == "gaussian_noise":
                        aug_data = self.add_gaussian_noise(aug_data)
                except Exception as e:
                    logger.warning(f"Augmentation '{t}' failed: {e}")

        return aug_data, aug_mask

    def random_rotation(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        max_angle: float = 15.0,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Random 3D rotation around a random axis."""
        angle = self.rng.uniform(-max_angle, max_angle)
        axes = self.rng.choice([(0, 1), (0, 2), (1, 2)])

        rotated_data = ndimage.rotate(
            data, angle, axes=axes, reshape=False, order=1, mode="constant", cval=0
        )

        rotated_mask = None
        if mask is not None:
            rotated_mask = ndimage.rotate(
                mask, angle, axes=axes, reshape=False, order=0, mode="constant", cval=0
            )

        return rotated_data.astype(np.float32), rotated_mask

    def random_flip(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Random flip along a random axis."""
        axis = self.rng.choice([0, 1, 2])
        flipped_data = np.flip(data, axis=axis).copy()
        flipped_mask = np.flip(mask, axis=axis).copy() if mask is not None else None
        return flipped_data, flipped_mask

    def random_brightness(
        self,
        data: np.ndarray,
        brightness_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Random brightness adjustment (additive)."""
        if brightness_range is None:
            brightness_range = tuple(
                config.get("augmentation.brightness_range", [-0.1, 0.1])
            )

        factor = self.rng.uniform(*brightness_range)
        fg_mask = data > 0
        result = data.copy()
        result[fg_mask] = data[fg_mask] + factor * data[fg_mask].std()
        return result.astype(np.float32)

    def random_contrast(
        self,
        data: np.ndarray,
        contrast_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Random contrast adjustment (multiplicative)."""
        if contrast_range is None:
            contrast_range = tuple(
                config.get("augmentation.contrast_range", [0.9, 1.1])
            )

        factor = self.rng.uniform(*contrast_range)
        fg_mask = data > 0
        mean_val = data[fg_mask].mean()
        result = data.copy()
        result[fg_mask] = (data[fg_mask] - mean_val) * factor + mean_val
        return result.astype(np.float32)

    def random_gamma(
        self,
        data: np.ndarray,
        gamma_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Random gamma correction."""
        if gamma_range is None:
            gamma_range = tuple(
                config.get("augmentation.gamma_range", [0.7, 1.5])
            )

        gamma = self.rng.uniform(*gamma_range)
        fg_mask = data > 0
        result = data.copy()

        # Normalize to [0, 1] for gamma
        fg = data[fg_mask]
        d_min, d_max = fg.min(), fg.max()
        normalized = (fg - d_min) / (d_max - d_min + 1e-8)
        gamma_corrected = np.power(normalized, gamma)
        result[fg_mask] = gamma_corrected * (d_max - d_min) + d_min

        return result.astype(np.float32)

    def random_scaling(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        scale_range: Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Random isotropic scaling."""
        if scale_range is None:
            scale_range = tuple(
                config.get("augmentation.scale_range", [0.9, 1.1])
            )

        scale = self.rng.uniform(*scale_range)
        scaled_data = ndimage.zoom(data, scale, order=1, mode="constant", cval=0)

        # Resize back to original shape
        scaled_data = self._resize_to_shape(scaled_data, data.shape)

        scaled_mask = None
        if mask is not None:
            scaled_mask = ndimage.zoom(mask, scale, order=0, mode="constant", cval=0)
            scaled_mask = self._resize_to_shape(scaled_mask, mask.shape)

        return scaled_data.astype(np.float32), scaled_mask

    def elastic_deformation(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        alpha: float = 100.0,
        sigma: float = 10.0,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply elastic deformation to simulate tissue variability.
        
        Args:
            data: 3D volume
            mask: Optional segmentation mask
            alpha: Deformation magnitude
            sigma: Smoothness of the deformation field
        """
        shape = data.shape[:3]

        # Generate random displacement fields
        dx = gaussian_filter(self.rng.randn(*shape) * alpha, sigma)
        dy = gaussian_filter(self.rng.randn(*shape) * alpha, sigma)
        dz = gaussian_filter(self.rng.randn(*shape) * alpha, sigma)

        # Create coordinate grids
        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        indices = [
            np.clip(x + dx, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(z + dz, 0, shape[2] - 1),
        ]

        deformed_data = map_coordinates(data, indices, order=1, mode="constant")

        deformed_mask = None
        if mask is not None:
            deformed_mask = map_coordinates(
                mask, indices, order=0, mode="constant"
            ).astype(mask.dtype)

        return deformed_data.astype(np.float32), deformed_mask

    def add_gaussian_noise(
        self,
        data: np.ndarray,
        noise_std: float = 0.01,
    ) -> np.ndarray:
        """Add Gaussian noise to the volume."""
        fg_mask = data > 0
        noise = self.rng.randn(*data.shape) * noise_std * data[fg_mask].std()
        result = data.copy()
        result[fg_mask] += noise[fg_mask]
        return result.astype(np.float32)

    def _resize_to_shape(
        self,
        data: np.ndarray,
        target_shape: tuple,
    ) -> np.ndarray:
        """Resize or crop/pad data to match target shape."""
        result = np.zeros(target_shape, dtype=data.dtype)

        # Calculate slicing ranges
        slices_src = []
        slices_dst = []
        for s, t in zip(data.shape, target_shape):
            if s >= t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(0, t))
            else:
                start = (t - s) // 2
                slices_src.append(slice(0, s))
                slices_dst.append(slice(start, start + s))

        result[tuple(slices_dst)] = data[tuple(slices_src)]
        return result
