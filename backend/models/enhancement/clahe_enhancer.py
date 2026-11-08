"""
NeuroFusion AI - CLAHE Enhancement Model
Baseline enhancement using Contrast Limited Adaptive Histogram Equalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from backend.models.enhancement.base import BaseEnhancer
from backend.utils.gpu import InferenceTimer


class CLAHEEnhancer(BaseEnhancer):
    """
    CLAHE-based MRI enhancement (baseline model).
    
    This is a non-learned baseline that applies CLAHE for contrast enhancement.
    Wrapped as a PyTorch module for API consistency.
    """

    def __init__(self, clip_limit: float = 3.0, grid_size: int = 8):
        super().__init__(model_name="CLAHE")
        self.clip_limit = clip_limit
        self.grid_size = grid_size

        # Dummy parameter so PyTorch recognizes this as a module
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CLAHE to a batch of 2D slices.
        
        Args:
            x: Input tensor (B, 1, H, W) in [0, 1] range
            
        Returns:
            CLAHE-enhanced tensor
        """
        import cv2

        batch_size = x.shape[0]
        device = x.device
        result = torch.zeros_like(x)

        for i in range(batch_size):
            slice_np = x[i, 0].cpu().numpy()

            # Convert to uint8 for OpenCV CLAHE
            slice_uint8 = (slice_np * 255).clip(0, 255).astype(np.uint8)

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.grid_size, self.grid_size),
            )
            enhanced = clahe.apply(slice_uint8)

            # Convert back to float [0, 1]
            result[i, 0] = torch.from_numpy(
                enhanced.astype(np.float32) / 255.0
            ).to(device)

        return result

    def enhance_volume(
        self,
        volume: np.ndarray,
        batch_size: int = 16,
        overlap: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        CLAHE-enhanced volume processing (overridden for efficiency).
        
        Uses direct OpenCV CLAHE without GPU overhead.
        """
        import cv2

        logger.info(f"CLAHE enhancement: clip={self.clip_limit}, grid={self.grid_size}")

        enhanced = np.zeros_like(volume, dtype=np.float32)

        fg_mask = volume > 0
        v_min = volume[fg_mask].min() if np.any(fg_mask) else 0
        v_max = volume[fg_mask].max() if np.any(fg_mask) else 1

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.grid_size, self.grid_size),
        )

        with InferenceTimer("CLAHE Enhancement") as timer:
            for i in range(volume.shape[2]):
                slice_2d = volume[:, :, i]

                if np.any(slice_2d > 0):
                    # Normalize to [0, 255]
                    normalized = (
                        (slice_2d - v_min) / (v_max - v_min + 1e-8) * 255
                    ).clip(0, 255).astype(np.uint8)

                    enhanced_slice = clahe.apply(normalized)
                    enhanced[:, :, i] = (
                        enhanced_slice.astype(np.float32) / 255.0 *
                        (v_max - v_min) + v_min
                    )

        metadata = {
            "model": "CLAHE",
            "clip_limit": self.clip_limit,
            "grid_size": self.grid_size,
            "num_slices": volume.shape[2],
            "inference_time_sec": timer.result.total_time_sec,
            "peak_memory_gb": 0.0,
            "device": "CPU",
        }

        return enhanced, metadata
