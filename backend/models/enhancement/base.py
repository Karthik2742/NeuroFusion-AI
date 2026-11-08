"""
NeuroFusion AI - Enhancement Base Class
Abstract base for all MRI enhancement models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from backend.utils.gpu import get_device, InferenceTimer


class BaseEnhancer(ABC, nn.Module):
    """
    Abstract base class for all MRI enhancement models.
    
    Provides common interface for:
        - Loading/saving weights
        - Inference on 3D volumes (slice-by-slice or volumetric)
        - Benchmarking
    """

    def __init__(self, model_name: str = "BaseEnhancer"):
        super().__init__()
        self.model_name = model_name
        self.device = get_device()
        self._is_loaded = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhancement model.
        
        Args:
            x: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
            
        Returns:
            Enhanced tensor of same shape
        """
        pass

    def enhance_volume(
        self,
        volume: np.ndarray,
        batch_size: int = 4,
        overlap: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Enhance a full 3D MRI volume.
        
        Processes the volume slice-by-slice through the model.
        
        Args:
            volume: 3D numpy array (H, W, D)
            batch_size: Number of slices to process at once
            overlap: Overlap between batches for blending
            
        Returns:
            Enhanced volume and inference metadata
        """
        self.eval()
        self.to(self.device)

        enhanced = np.zeros_like(volume, dtype=np.float32)
        num_slices = volume.shape[2]

        # Normalize input to [0, 1]
        fg_mask = volume > 0
        v_min = volume[fg_mask].min() if np.any(fg_mask) else 0
        v_max = volume[fg_mask].max() if np.any(fg_mask) else 1
        normalized = np.where(
            fg_mask,
            (volume - v_min) / (v_max - v_min + 1e-8),
            0,
        )

        with InferenceTimer(f"Enhancement ({self.model_name})") as timer:
            with torch.no_grad():
                for start in range(0, num_slices, batch_size):
                    end = min(start + batch_size, num_slices)

                    # Extract batch of slices
                    batch_slices = []
                    for i in range(start, end):
                        slice_2d = normalized[:, :, i]
                        batch_slices.append(slice_2d)

                    # Stack into batch tensor (B, 1, H, W)
                    batch_np = np.stack(batch_slices, axis=0)
                    batch_tensor = torch.from_numpy(batch_np).unsqueeze(1).float()
                    batch_tensor = batch_tensor.to(self.device)

                    # Forward pass
                    output = self.forward(batch_tensor)
                    output_np = output.cpu().numpy()
                    if output_np.ndim == 4:
                        output_np = output_np[:, 0, :, :]

                    # Store results
                    for i, slice_idx in enumerate(range(start, end)):
                        enhanced[:, :, slice_idx] = output_np[i]

        # Scale back to original range
        enh_fg = enhanced > 0
        if np.any(enh_fg):
            enhanced[enh_fg] = enhanced[enh_fg] * (v_max - v_min) + v_min

        metadata = {
            "model": self.model_name,
            "num_slices": num_slices,
            "batch_size": batch_size,
            "inference_time_sec": timer.result.total_time_sec,
            "peak_memory_gb": timer.result.peak_memory_gb,
            "device": timer.result.device,
        }

        logger.info(
            f"Volume enhancement complete: {self.model_name} | "
            f"{timer.result.total_time_sec:.2f}s | "
            f"{timer.result.peak_memory_gb:.3f} GB"
        )

        return enhanced, metadata

    def load_weights(self, filepath: str | Path) -> None:
        """Load pre-trained weights from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Weights file not found: {filepath}")
            return

        state_dict = torch.load(str(filepath), map_location=self.device)
        if "model_state_dict" in state_dict:
            self.load_state_dict(state_dict["model_state_dict"])
        elif "state_dict" in state_dict:
            self.load_state_dict(state_dict["state_dict"])
        else:
            self.load_state_dict(state_dict)

        self._is_loaded = True
        logger.info(f"Loaded weights: {filepath.name}")

    def save_weights(self, filepath: str | Path) -> None:
        """Save model weights to a file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
        }, str(filepath))
        logger.info(f"Saved weights: {filepath.name}")

    def get_num_parameters(self) -> dict[str, int]:
        """Get model parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }

    def __repr__(self) -> str:
        params = self.get_num_parameters()
        return (
            f"{self.model_name}("
            f"params={params['total']:,}, "
            f"trainable={params['trainable']:,})"
        )
