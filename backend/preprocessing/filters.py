"""
NeuroFusion AI - Preprocessing Filters
Image filtering and enhancement operations for MRI volumes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from scipy import ndimage
from scipy.ndimage import uniform_filter, generic_filter

from backend.configs import config


class MRIFilters:
    """
    Collection of image filtering operations optimized for MRI data.
    
    All filters operate on 3D volumes and preserve the data type.
    """

    @staticmethod
    def clahe_3d(
        data: np.ndarray,
        clip_limit: float = 3.0,
        grid_size: int = 8,
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on 3D MRI.
        
        Since CLAHE is natively 2D (OpenCV), we apply it slice-by-slice
        across the axial plane for 3D volumes.
        
        Args:
            data: 3D volume
            clip_limit: Contrast limiting threshold
            grid_size: Size of the grid for histogram equalization
            
        Returns:
            CLAHE-enhanced volume
        """
        import cv2

        if clip_limit is None:
            clip_limit = config.get("preprocessing.clahe.clip_limit", 3.0)

        result = np.zeros_like(data, dtype=np.float32)

        # Normalize to [0, 255] for CLAHE
        non_zero = data > 0
        if not np.any(non_zero):
            return data

        d_min = data[non_zero].min()
        d_max = data[non_zero].max()
        normalized = np.zeros_like(data, dtype=np.uint8)
        normalized[non_zero] = (
            (data[non_zero] - d_min) / (d_max - d_min + 1e-8) * 255
        ).astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size),
        )

        # Apply CLAHE slice-by-slice (axial)
        for i in range(data.shape[2]):
            slice_2d = normalized[:, :, i]
            if np.any(slice_2d > 0):
                enhanced_slice = clahe.apply(slice_2d)
                result[:, :, i] = enhanced_slice.astype(np.float32)
            else:
                result[:, :, i] = 0.0

        # Scale back to original range
        result_non_zero = result > 0
        if np.any(result_non_zero):
            r_min = result[result_non_zero].min()
            r_max = result[result_non_zero].max()
            result[result_non_zero] = (
                (result[result_non_zero] - r_min) / (r_max - r_min + 1e-8)
                * (d_max - d_min) + d_min
            )

        logger.info(f"CLAHE applied: clip_limit={clip_limit}, grid={grid_size}")
        return result

    @staticmethod
    def histogram_equalization(data: np.ndarray) -> np.ndarray:
        """
        Apply global histogram equalization to a 3D volume.
        
        Args:
            data: 3D volume
            
        Returns:
            Equalized volume
        """
        import cv2

        result = np.zeros_like(data, dtype=np.float32)
        non_zero = data > 0
        d_min = data[non_zero].min() if np.any(non_zero) else 0
        d_max = data[non_zero].max() if np.any(non_zero) else 1

        for i in range(data.shape[2]):
            slice_2d = data[:, :, i]
            if np.any(slice_2d > 0):
                normalized = (
                    (slice_2d - d_min) / (d_max - d_min + 1e-8) * 255
                ).astype(np.uint8)
                equalized = cv2.equalizeHist(normalized)
                result[:, :, i] = equalized.astype(np.float32)

        # Scale back
        if np.any(result > 0):
            result_nz = result > 0
            r_min, r_max = result[result_nz].min(), result[result_nz].max()
            result[result_nz] = (
                (result[result_nz] - r_min) / (r_max - r_min + 1e-8)
                * (d_max - d_min) + d_min
            )

        logger.info("Histogram equalization applied")
        return result

    @staticmethod
    def median_filter(
        data: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Apply 3D median filter for noise removal.
        
        Preserves edges while removing salt-and-pepper noise.
        
        Args:
            data: 3D volume
            kernel_size: Filter kernel size
            
        Returns:
            Filtered volume
        """
        if kernel_size is None:
            kernel_size = config.get("preprocessing.median_filter.kernel_size", 3)

        result = ndimage.median_filter(data, size=kernel_size)
        logger.info(f"Median filter applied: kernel={kernel_size}")
        return result.astype(np.float32)

    @staticmethod
    def gaussian_filter(
        data: np.ndarray,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Apply 3D Gaussian smoothing filter.
        
        Args:
            data: 3D volume
            sigma: Standard deviation of the Gaussian kernel
            
        Returns:
            Smoothed volume
        """
        if sigma is None:
            sigma = config.get("preprocessing.gaussian_filter.sigma", 1.0)

        result = ndimage.gaussian_filter(data, sigma=sigma)
        logger.info(f"Gaussian filter applied: sigma={sigma}")
        return result.astype(np.float32)

    @staticmethod
    def non_local_means(
        data: np.ndarray,
        patch_size: int = 5,
        patch_distance: int = 6,
        h: float = 0.1,
    ) -> np.ndarray:
        """
        Apply Non-Local Means denoising.
        
        Uses self-similarity of patches to denoise while preserving structure.
        Applied slice-by-slice for memory efficiency on 3D volumes.
        
        Args:
            data: 3D volume
            patch_size: Size of patches for comparison
            patch_distance: Maximal distance for patch search
            h: Cut-off distance (higher = more smoothing)
            
        Returns:
            Denoised volume
        """
        from skimage.restoration import denoise_nl_means, estimate_sigma

        result = np.zeros_like(data, dtype=np.float32)

        # Estimate noise level
        non_zero = data[data > 0]
        sigma_est = float(np.std(data[data < np.percentile(non_zero, 5)])) if len(non_zero) > 0 else 0.01

        for i in range(data.shape[2]):
            slice_2d = data[:, :, i].astype(np.float64)
            if np.any(slice_2d > 0):
                denoised = denoise_nl_means(
                    slice_2d,
                    patch_size=patch_size,
                    patch_distance=patch_distance,
                    h=h * sigma_est,
                    fast_mode=True,
                )
                result[:, :, i] = denoised.astype(np.float32)

        logger.info(
            f"Non-Local Means applied: patch={patch_size}, "
            f"distance={patch_distance}, h={h}"
        )
        return result

    @staticmethod
    def anisotropic_diffusion(
        data: np.ndarray,
        n_iterations: int = 10,
        kappa: float = 50.0,
        gamma: float = 0.1,
        option: int = 1,
    ) -> np.ndarray:
        """
        Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.
        
        Args:
            data: 3D volume
            n_iterations: Number of diffusion iterations
            kappa: Conduction coefficient (controls edge sensitivity)
            gamma: Speed of diffusion (0 < gamma <= 0.25 for stability)
            option: 1 = exponential, 2 = quadratic diffusion function
            
        Returns:
            Diffused volume
        """
        result = data.astype(np.float64).copy()

        for _ in range(n_iterations):
            # Compute gradients in all directions
            delta_n = np.zeros_like(result)
            delta_s = np.zeros_like(result)
            delta_e = np.zeros_like(result)
            delta_w = np.zeros_like(result)
            delta_u = np.zeros_like(result)
            delta_d = np.zeros_like(result)

            delta_n[:-1, :, :] = result[1:, :, :] - result[:-1, :, :]
            delta_s[1:, :, :] = result[:-1, :, :] - result[1:, :, :]
            delta_e[:, :-1, :] = result[:, 1:, :] - result[:, :-1, :]
            delta_w[:, 1:, :] = result[:, :-1, :] - result[:, 1:, :]
            delta_u[:, :, :-1] = result[:, :, 1:] - result[:, :, :-1]
            delta_d[:, :, 1:] = result[:, :, :-1] - result[:, :, 1:]

            # Compute conduction coefficients
            if option == 1:
                # Exponential
                cn = np.exp(-(delta_n / kappa) ** 2)
                cs = np.exp(-(delta_s / kappa) ** 2)
                ce = np.exp(-(delta_e / kappa) ** 2)
                cw = np.exp(-(delta_w / kappa) ** 2)
                cu = np.exp(-(delta_u / kappa) ** 2)
                cd = np.exp(-(delta_d / kappa) ** 2)
            else:
                # Quadratic
                cn = 1.0 / (1.0 + (delta_n / kappa) ** 2)
                cs = 1.0 / (1.0 + (delta_s / kappa) ** 2)
                ce = 1.0 / (1.0 + (delta_e / kappa) ** 2)
                cw = 1.0 / (1.0 + (delta_w / kappa) ** 2)
                cu = 1.0 / (1.0 + (delta_u / kappa) ** 2)
                cd = 1.0 / (1.0 + (delta_d / kappa) ** 2)

            # Update
            result += gamma * (
                cn * delta_n + cs * delta_s +
                ce * delta_e + cw * delta_w +
                cu * delta_u + cd * delta_d
            )

        logger.info(
            f"Anisotropic diffusion applied: iterations={n_iterations}, "
            f"kappa={kappa}, gamma={gamma}"
        )
        return result.astype(np.float32)
