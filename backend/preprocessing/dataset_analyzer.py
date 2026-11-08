"""
NeuroFusion AI - Module 1: Dataset Analyzer
Automatically detects MRI type, modality, and generates comprehensive dataset statistics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
from loguru import logger
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy

from backend.utils.io import NIfTILoader
from backend.utils.visualization import (
    generate_histogram,
    generate_metrics_chart,
    generate_slice_views,
    volume_to_slice_images,
    fig_to_base64,
    apply_neurofusion_style,
    NEUROFUSION_COLORS,
    _array_to_base64_png,
)
import cv2


class DatasetAnalyzer:
    """
    Comprehensive MRI dataset analysis engine.
    
    Capabilities:
        - Brain MRI volumetric analysis
        - Identify MRI modality (T1, T1c, T2, FLAIR)
        - Generate volumetric statistics
        - Estimate image quality metrics
        - Generate publication-quality charts
    """

    # Typical intensity distribution characteristics for modality detection
    BRAIN_MODALITY_SIGNATURES = {
        "t1": {"csf_dark": True, "gm_intermediate": True, "wm_bright": True},
        "t1ce": {"csf_dark": True, "gm_intermediate": True, "wm_bright": True, "enhancement": True},
        "t2": {"csf_bright": True, "gm_intermediate": True, "wm_dark": True},
        "flair": {"csf_dark": True, "gm_intermediate": True, "wm_bright": True, "lesions_bright": True},
    }

    def __init__(self):
        self.loader = NIfTILoader()

    def analyze_file(self, filepath: str | Path) -> dict[str, Any]:
        """
        Perform comprehensive analysis on a single NIfTI file.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            Complete analysis dictionary with statistics, quality metrics, and charts
        """
        filepath = Path(filepath)
        logger.info(f"Analyzing: {filepath.name}")

        # Load the volume
        data, header, affine = self.loader.load_nibabel(filepath)

        # Core analysis
        metadata = self.loader.get_metadata(filepath)
        quality_metrics = self._compute_quality_metrics(data)
        intensity_stats = self._compute_intensity_stats(data)
        spatial_stats = self._compute_spatial_stats(data, header, affine)
        mri_type = self._detect_mri_type(data, header)
        modality = self._detect_modality(data, filepath.name)

        # Generate visualizations
        charts = self._generate_charts(data, filepath.name)

        result = {
            "file_info": metadata,
            "mri_type": mri_type,
            "modality": modality,
            "intensity_stats": intensity_stats,
            "spatial_stats": spatial_stats,
            "quality_metrics": quality_metrics,
            "charts": charts,
        }

        logger.info(
            f"Analysis complete: type={mri_type}, modality={modality}, "
            f"shape={data.shape}"
        )
        return result

    def analyze_dataset(self, dataset_dir: str | Path) -> dict[str, Any]:
        """
        Analyze an entire dataset directory.
        
        Args:
            dataset_dir: Root directory containing NIfTI files or patient folders
            
        Returns:
            Dataset-level analysis with per-file results and aggregated stats
        """
        dataset_dir = Path(dataset_dir)
        logger.info(f"Analyzing dataset: {dataset_dir}")

        nifti_files = self._find_nifti_files(dataset_dir)
        logger.info(f"Found {len(nifti_files)} NIfTI files")

        if not nifti_files:
            return {"error": "No NIfTI files found", "path": str(dataset_dir)}

        file_analyses = []
        all_shapes = []
        all_spacings = []
        modality_counts = {}
        type_counts = {"brain": 0, "unknown": 0}

        for i, filepath in enumerate(nifti_files[:100]):
            try:
                analysis = self.analyze_file(filepath)
                file_analyses.append(analysis)

                all_shapes.append(analysis["file_info"]["shape"][:3])
                all_spacings.append(analysis["spatial_stats"]["voxel_spacing"])

                mri_type = analysis["mri_type"]
                type_counts[mri_type] = type_counts.get(mri_type, 0) + 1

                modality = analysis["modality"]
                modality_counts[modality] = modality_counts.get(modality, 0) + 1

            except Exception as e:
                logger.warning(f"Failed to analyze {filepath}: {e}")

        dataset_stats = {
            "total_files": len(nifti_files),
            "analyzed_files": len(file_analyses),
            "type_distribution": type_counts,
            "modality_distribution": modality_counts,
            "shape_stats": self._aggregate_shapes(all_shapes),
            "spacing_stats": self._aggregate_spacings(all_spacings),
            "dataset_path": str(dataset_dir),
        }

        dataset_charts = self._generate_dataset_charts(
            dataset_stats, file_analyses
        )

        return {
            "dataset_stats": dataset_stats,
            "dataset_charts": dataset_charts,
            "file_analyses": file_analyses[:10],
        }

    def _detect_mri_type(self, data: np.ndarray, header: nib.Nifti1Header) -> str:
        """Detect MRI type (Brain MRI platform)."""
        return "brain"

    def _detect_modality(self, data: np.ndarray, filename: str) -> str:
        """Detect MRI modality from filename patterns and intensity characteristics."""
        filename_lower = filename.lower()

        if "flair" in filename_lower:
            return "FLAIR"
        if "t1ce" in filename_lower or "t1gd" in filename_lower or "t1c" in filename_lower:
            return "T1c"
        if "t1" in filename_lower:
            return "T1"
        if "t2" in filename_lower:
            return "T2"
        if "seg" in filename_lower or "mask" in filename_lower or "label" in filename_lower:
            return "Segmentation"

        return self._detect_modality_from_intensity(data)

    def _detect_modality_from_intensity(self, data: np.ndarray) -> str:
        """Detect MRI modality purely from intensity distribution analysis."""
        non_zero = data[data > 0].flatten()
        if len(non_zero) < 1000:
            return "Unknown"

        normalized = (non_zero - non_zero.min()) / (non_zero.max() - non_zero.min() + 1e-8)
        hist, bin_edges = np.histogram(normalized, bins=100, density=True)
        peak_position = bin_edges[np.argmax(hist)]
        skewness = float(np.mean(((normalized - normalized.mean()) / (normalized.std() + 1e-8)) ** 3))

        if peak_position < 0.3 and skewness > 1.0:
            return "T2"
        elif peak_position > 0.3 and skewness < 0.5:
            return "T1"
        elif 0.2 < peak_position < 0.5 and skewness > 0.3:
            return "FLAIR"
        else:
            return "T1"

    def _compute_intensity_stats(self, data: np.ndarray) -> dict[str, Any]:
        """Compute comprehensive intensity statistics."""
        non_zero = data[data > 0]
        all_data = data.flatten()

        return {
            "global": {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "median": float(np.median(data)),
            },
            "foreground": {
                "min": float(np.min(non_zero)) if len(non_zero) > 0 else 0,
                "max": float(np.max(non_zero)) if len(non_zero) > 0 else 0,
                "mean": float(np.mean(non_zero)) if len(non_zero) > 0 else 0,
                "std": float(np.std(non_zero)) if len(non_zero) > 0 else 0,
                "median": float(np.median(non_zero)) if len(non_zero) > 0 else 0,
            },
            "percentiles": {
                "p1": float(np.percentile(all_data, 1)),
                "p5": float(np.percentile(all_data, 5)),
                "p25": float(np.percentile(all_data, 25)),
                "p50": float(np.percentile(all_data, 50)),
                "p75": float(np.percentile(all_data, 75)),
                "p95": float(np.percentile(all_data, 95)),
                "p99": float(np.percentile(all_data, 99)),
            },
            "non_zero_ratio": float(np.count_nonzero(data) / data.size),
        }

    def _compute_spatial_stats(
        self, data: np.ndarray, header: nib.Nifti1Header, affine: np.ndarray
    ) -> dict[str, Any]:
        """Compute spatial/geometric statistics."""
        zooms = header.get_zooms()
        orientation = "".join(nib.aff2axcodes(affine))

        non_zero = np.argwhere(data > 0)
        if len(non_zero) > 0:
            bbox_min = non_zero.min(axis=0).tolist()
            bbox_max = non_zero.max(axis=0).tolist()
            extent = [int(mx - mn + 1) for mn, mx in zip(bbox_min, bbox_max)]
        else:
            bbox_min = [0, 0, 0]
            bbox_max = list(data.shape[:3])
            extent = list(data.shape[:3])

        physical_size = [
            round(float(s * z), 2) for s, z in zip(data.shape[:3], zooms[:3])
        ]

        return {
            "shape": list(data.shape),
            "voxel_spacing": [round(float(z), 4) for z in zooms[:3]],
            "orientation": orientation,
            "physical_size_mm": physical_size,
            "bounding_box": {"min": bbox_min, "max": bbox_max},
            "foreground_extent": extent,
            "total_voxels": int(np.prod(data.shape[:3])),
            "volume_ml": round(
                float(np.count_nonzero(data) * np.prod(zooms[:3])) / 1000, 2
            ),
        }

    def _compute_quality_metrics(self, data: np.ndarray) -> dict[str, float]:
        """Compute image quality metrics."""
        non_zero = data[data > 0]
        if len(non_zero) < 100:
            return {}

        normalized = non_zero / (non_zero.max() + 1e-8)

        bg_mask = data < np.percentile(data[data > 0], 5)
        noise_std = float(np.std(data[bg_mask])) if np.any(bg_mask) else 0.0

        signal_mean = float(np.mean(non_zero))
        snr = signal_mean / (noise_std + 1e-8)

        mid_slice = data[:, :, data.shape[2] // 2]
        laplacian = ndimage.laplace(mid_slice.astype(np.float64))
        sharpness = float(np.var(laplacian))

        brightness = float(np.mean(normalized))
        contrast = float(np.std(normalized))

        hist, _ = np.histogram(non_zero, bins=256, density=True)
        hist = hist[hist > 0]
        image_entropy = float(scipy_entropy(hist, base=2))

        p25 = np.percentile(non_zero, 25)
        p75 = np.percentile(non_zero, 75)
        cnr = float((p75 - p25) / (noise_std + 1e-8))

        return {
            "noise_estimate": round(noise_std, 4),
            "snr": round(snr, 2),
            "cnr": round(cnr, 2),
            "sharpness": round(sharpness, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "entropy": round(image_entropy, 4),
        }

    def _generate_charts(self, data: np.ndarray, filename: str) -> dict[str, str]:
        """Generate visualization charts for a single file."""
        charts = {}
        charts["histogram"] = generate_histogram(data, title=f"Intensity Distribution — {filename}")
        charts["slice_views"] = generate_slice_views(data, title=f"Multi-plane Views — {filename}")
        charts["slice_grid"] = volume_to_slice_images(data, num_slices=9, title=f"Slice Grid — {filename}")
        return charts

    def _generate_dataset_charts(
        self,
        dataset_stats: dict,
        file_analyses: list[dict],
    ) -> dict[str, str]:
        """Generate dataset-level summary charts."""
        charts = {}

        mod_dist = dataset_stats.get("modality_distribution", {})
        if mod_dist:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                fig.patch.set_facecolor(NEUROFUSION_COLORS["background"])
                colors = [
                    NEUROFUSION_COLORS["primary"],
                    NEUROFUSION_COLORS["secondary"],
                    NEUROFUSION_COLORS["accent"],
                    NEUROFUSION_COLORS["warning"],
                    NEUROFUSION_COLORS["danger"],
                ]
                wedges, texts, autotexts = ax.pie(
                    list(mod_dist.values()),
                    labels=list(mod_dist.keys()),
                    colors=colors[:len(mod_dist)],
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": NEUROFUSION_COLORS["text"], "fontsize": 11},
                )
                for t in autotexts:
                    t.set_fontsize(10)
                    t.set_fontweight("bold")
                ax.set_title("Modality Distribution", fontsize=14, color=NEUROFUSION_COLORS["primary"], fontweight="bold")
                charts["modality_distribution"] = fig_to_base64(fig)
            except Exception:
                canvas = np.full((120, 600, 3), (15, 20, 50), dtype=np.uint8)
                cv2.putText(canvas, f"Modality Distribution: {mod_dist}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
                charts["modality_distribution"] = _array_to_base64_png(canvas)

        type_dist = dataset_stats.get("type_distribution", {})
        if type_dist:
            charts["type_distribution"] = generate_metrics_chart(
                {k: v for k, v in type_dist.items() if v > 0},
                title="MRI Type Distribution",
                chart_type="bar",
            )

        if file_analyses:
            avg_metrics = {}
            metric_keys = ["snr", "cnr", "sharpness", "entropy", "contrast"]
            for key in metric_keys:
                values = [
                    fa["quality_metrics"].get(key, 0)
                    for fa in file_analyses
                    if key in fa.get("quality_metrics", {})
                ]
                if values:
                    avg_metrics[key.upper()] = np.mean(values)

            if avg_metrics:
                charts["avg_quality_metrics"] = generate_metrics_chart(
                    avg_metrics,
                    title="Average Quality Metrics",
                    chart_type="horizontal",
                )

        return charts

    def _find_nifti_files(self, directory: Path) -> list[Path]:
        """Recursively find all NIfTI files in a directory."""
        nifti_files = []
        for ext in ["*.nii", "*.nii.gz"]:
            nifti_files.extend(directory.rglob(ext))
        return sorted(nifti_files)

    def _aggregate_shapes(self, shapes: list[list[int]]) -> dict[str, Any]:
        """Aggregate volume shape statistics."""
        if not shapes:
            return {}
        arr = np.array(shapes)
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).round(1).tolist(),
            "unique_shapes": len(set(tuple(s) for s in shapes)),
        }

    def _aggregate_spacings(self, spacings: list[list[float]]) -> dict[str, Any]:
        """Aggregate voxel spacing statistics."""
        if not spacings:
            return {}
        arr = np.array(spacings)
        return {
            "min": arr.min(axis=0).round(4).tolist(),
            "max": arr.max(axis=0).round(4).tolist(),
            "mean": arr.mean(axis=0).round(4).tolist(),
            "is_isotropic": bool(np.allclose(arr.mean(axis=0), arr.mean(axis=0)[0], atol=0.01)),
        }
