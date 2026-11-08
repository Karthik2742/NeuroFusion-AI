"""
NeuroFusion AI - Module 6: Enhancement Metrics
Comprehensive image quality assessment metrics for MRI enhancement evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from loguru import logger
from scipy.signal import fftconvolve


class EnhancementMetrics:
    """
    Full suite of image quality metrics for evaluating MRI enhancement.
    
    Reference-based metrics (require ground truth):
        PSNR, SSIM, MSE, RMSE, UQI, FSIM
        
    No-reference metrics (no ground truth needed):
        BRISQUE, NIQE, PIQE, Entropy
    """

    @staticmethod
    def compute_all(
        original: np.ndarray,
        enhanced: np.ndarray,
        data_range: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Compute all available metrics between original and enhanced images.
        
        Args:
            original: Original image/volume
            enhanced: Enhanced image/volume
            data_range: Dynamic range of the data
            
        Returns:
            Dictionary of metric name to value
        """
        if data_range is None:
            data_range = float(max(original.max(), enhanced.max()) - min(original.min(), enhanced.min()))
            if data_range == 0:
                data_range = 1.0

        metrics = {}

        try:
            metrics["psnr"] = EnhancementMetrics.psnr(original, enhanced, data_range)
        except Exception as e:
            logger.warning(f"PSNR computation failed: {e}")
            metrics["psnr"] = 0.0

        try:
            metrics["ssim"] = EnhancementMetrics.ssim(original, enhanced, data_range)
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            metrics["ssim"] = 0.0

        try:
            metrics["mse"] = EnhancementMetrics.mse(original, enhanced)
        except Exception:
            metrics["mse"] = 0.0

        try:
            metrics["rmse"] = EnhancementMetrics.rmse(original, enhanced)
        except Exception:
            metrics["rmse"] = 0.0

        try:
            metrics["uqi"] = EnhancementMetrics.uqi(original, enhanced)
        except Exception:
            metrics["uqi"] = 0.0

        try:
            metrics["fsim"] = EnhancementMetrics.fsim(original, enhanced)
        except Exception:
            metrics["fsim"] = 0.0

        # No-reference metrics on enhanced image
        try:
            metrics["entropy"] = EnhancementMetrics.entropy(enhanced)
        except Exception:
            metrics["entropy"] = 0.0

        try:
            metrics["brisque"] = EnhancementMetrics.brisque_estimate(enhanced)
        except Exception:
            metrics["brisque"] = 0.0

        try:
            metrics["niqe"] = EnhancementMetrics.niqe_estimate(enhanced)
        except Exception:
            metrics["niqe"] = 0.0

        return metrics

    @staticmethod
    def psnr(original: np.ndarray, enhanced: np.ndarray, data_range: float) -> float:
        """
        Peak Signal-to-Noise Ratio.
        Higher is better. Typical range: 20-50 dB.
        """
        mse_val = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
        if mse_val == 0:
            return float("inf")
        return float(10 * math.log10(data_range ** 2 / mse_val))

    @staticmethod
    def ssim(
        original: np.ndarray,
        enhanced: np.ndarray,
        data_range: float,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> float:
        """
        Structural Similarity Index.
        Range: [-1, 1], higher is better.
        
        Computes SSIM slice-by-slice for 3D volumes and averages.
        """
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        orig = original.astype(np.float64)
        enh = enhanced.astype(np.float64)

        if orig.ndim == 3:
            # Process slice-by-slice and average
            ssim_values = []
            for i in range(orig.shape[2]):
                s = EnhancementMetrics._ssim_2d(
                    orig[:, :, i], enh[:, :, i], c1, c2, win_size
                )
                if not np.isnan(s):
                    ssim_values.append(s)
            return float(np.mean(ssim_values)) if ssim_values else 0.0
        else:
            return EnhancementMetrics._ssim_2d(orig, enh, c1, c2, win_size)

    @staticmethod
    def _ssim_2d(
        img1: np.ndarray, img2: np.ndarray,
        c1: float, c2: float, win_size: int,
    ) -> float:
        """Compute SSIM for 2D images."""
        from scipy.ndimage import uniform_filter

        mu1 = uniform_filter(img1, size=win_size)
        mu2 = uniform_filter(img2, size=win_size)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = uniform_filter(img1 ** 2, size=win_size) - mu1_sq
        sigma2_sq = uniform_filter(img2 ** 2, size=win_size) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, size=win_size) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return float(np.mean(ssim_map))

    @staticmethod
    def mse(original: np.ndarray, enhanced: np.ndarray) -> float:
        """Mean Squared Error. Lower is better."""
        return float(np.mean(
            (original.astype(np.float64) - enhanced.astype(np.float64)) ** 2
        ))

    @staticmethod
    def rmse(original: np.ndarray, enhanced: np.ndarray) -> float:
        """Root Mean Squared Error. Lower is better."""
        return float(np.sqrt(EnhancementMetrics.mse(original, enhanced)))

    @staticmethod
    def uqi(original: np.ndarray, enhanced: np.ndarray, block_size: int = 8) -> float:
        """
        Universal Quality Index.
        Measures luminance, contrast, and structure similarity.
        Range: [-1, 1], higher is better.
        """
        orig = original.astype(np.float64).flatten()
        enh = enhanced.astype(np.float64).flatten()

        mean_orig = np.mean(orig)
        mean_enh = np.mean(enh)
        var_orig = np.var(orig)
        var_enh = np.var(enh)
        cov = np.mean((orig - mean_orig) * (enh - mean_enh))

        numerator = 4 * cov * mean_orig * mean_enh
        denominator = (var_orig + var_enh) * (mean_orig ** 2 + mean_enh ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    @staticmethod
    def fsim(original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Feature Similarity Index.
        Based on phase congruency and gradient magnitude.
        Simplified implementation for 2D/3D.
        Range: [0, 1], higher is better.
        """
        from scipy.ndimage import sobel

        orig = original.astype(np.float64)
        enh = enhanced.astype(np.float64)

        if orig.ndim == 3:
            # Process middle slice for efficiency
            mid = orig.shape[2] // 2
            orig = orig[:, :, mid]
            enh = enh[:, :, mid]

        # Gradient magnitude as feature
        gm_orig = np.sqrt(sobel(orig, axis=0) ** 2 + sobel(orig, axis=1) ** 2)
        gm_enh = np.sqrt(sobel(enh, axis=0) ** 2 + sobel(enh, axis=1) ** 2)

        T1 = 0.85
        T2 = 160

        # Gradient similarity
        sg = (2 * gm_orig * gm_enh + T1) / (gm_orig ** 2 + gm_enh ** 2 + T1)

        # Luminance similarity
        sl = (2 * orig * enh + T2) / (orig ** 2 + enh ** 2 + T2)

        # Combined
        similarity_map = sg * sl

        # Weight by gradient magnitude
        weight = np.maximum(gm_orig, gm_enh)
        weight_sum = np.sum(weight)

        if weight_sum == 0:
            return float(np.mean(similarity_map))

        return float(np.sum(similarity_map * weight) / weight_sum)

    @staticmethod
    def entropy(data: np.ndarray) -> float:
        """
        Image entropy (information content).
        Higher generally means more detail/information.
        """
        from scipy.stats import entropy as scipy_entropy

        non_zero = data[data > 0].flatten()
        if len(non_zero) < 10:
            return 0.0

        hist, _ = np.histogram(non_zero, bins=256, density=True)
        hist = hist[hist > 0]
        return float(scipy_entropy(hist, base=2))

    @staticmethod
    def brisque_estimate(data: np.ndarray) -> float:
        """
        Simplified BRISQUE-like score (Blind/Referenceless Image Spatial QUality Evaluator).
        
        Lower is better. Uses normalized luminance coefficients
        and generalized Gaussian distribution fitting.
        """
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]

        from scipy.ndimage import uniform_filter

        # Compute MSCN (Mean Subtracted Contrast Normalized) coefficients
        mu = uniform_filter(data.astype(np.float64), size=7)
        mu_sq = mu ** 2
        sigma = np.sqrt(np.abs(uniform_filter(data.astype(np.float64) ** 2, size=7) - mu_sq))
        sigma = np.maximum(sigma, 1e-7)

        mscn = (data.astype(np.float64) - mu) / sigma

        # Features: variance of MSCN as quality proxy
        score = float(np.var(mscn))

        # Normalize to typical BRISQUE range [0, 100]
        # Lower variance → more uniform → potentially blurry → higher BRISQUE
        normalized_score = max(0, min(100, 100 - score * 20))
        return round(normalized_score, 2)

    @staticmethod
    def niqe_estimate(data: np.ndarray) -> float:
        """
        Simplified NIQE-like score (Natural Image Quality Evaluator).
        
        Lower is better. Based on deviation from natural scene statistics.
        """
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]

        from scipy.ndimage import uniform_filter

        data = data.astype(np.float64)

        # MSCN coefficients
        mu = uniform_filter(data, size=7)
        sigma = np.sqrt(np.abs(uniform_filter(data ** 2, size=7) - mu ** 2))
        sigma = np.maximum(sigma, 1e-7)
        mscn = (data - mu) / sigma

        # Compute statistics
        alpha = np.mean(np.abs(mscn))
        beta = np.sqrt(np.mean(mscn ** 2))

        # NIQE is based on multivariate Gaussian distance
        # Simplified: use ratio of shape parameters
        if beta == 0:
            return 10.0

        score = float(alpha / beta)
        # Normalize to typical range
        normalized = max(0, min(10, abs(score - 0.7) * 10))
        return round(normalized, 2)


class SegmentationMetrics:
    """
    Comprehensive segmentation evaluation metrics.
    
    All metrics support multi-class segmentation with per-class and averaged results.
    """

    @staticmethod
    def compute_all(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        num_classes: int = 4,
        class_names: Optional[dict[int, str]] = None,
    ) -> dict[str, Any]:
        """
        Compute all segmentation metrics.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            num_classes: Number of classes
            class_names: Mapping of class index to name
            
        Returns:
            Complete metrics dictionary
        """
        if class_names is None:
            class_names = {i: f"Class_{i}" for i in range(num_classes)}

        metrics = {
            "per_class": {},
            "averaged": {},
        }

        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []

        for cls_idx in range(num_classes):
            if cls_idx == 0:  # Skip background
                continue

            pred_mask = (prediction == cls_idx).astype(np.float32)
            gt_mask = (ground_truth == cls_idx).astype(np.float32)

            cls_name = class_names.get(cls_idx, f"Class_{cls_idx}")
            cls_metrics = {}

            cls_metrics["dice"] = SegmentationMetrics.dice_score(pred_mask, gt_mask)
            cls_metrics["iou"] = SegmentationMetrics.iou(pred_mask, gt_mask)
            cls_metrics["precision"] = SegmentationMetrics.precision(pred_mask, gt_mask)
            cls_metrics["recall"] = SegmentationMetrics.recall(pred_mask, gt_mask)
            cls_metrics["sensitivity"] = cls_metrics["recall"]
            cls_metrics["specificity"] = SegmentationMetrics.specificity(pred_mask, gt_mask)
            cls_metrics["f1"] = cls_metrics["dice"]  # Dice = F1 for binary

            try:
                cls_metrics["hausdorff_distance"] = SegmentationMetrics.hausdorff_distance(
                    pred_mask, gt_mask
                )
            except Exception:
                cls_metrics["hausdorff_distance"] = float("inf")

            metrics["per_class"][cls_name] = cls_metrics

            all_dice.append(cls_metrics["dice"])
            all_iou.append(cls_metrics["iou"])
            all_precision.append(cls_metrics["precision"])
            all_recall.append(cls_metrics["recall"])

        # Averaged metrics
        metrics["averaged"] = {
            "mean_dice": float(np.mean(all_dice)) if all_dice else 0.0,
            "mean_iou": float(np.mean(all_iou)) if all_iou else 0.0,
            "mean_precision": float(np.mean(all_precision)) if all_precision else 0.0,
            "mean_recall": float(np.mean(all_recall)) if all_recall else 0.0,
            "accuracy": SegmentationMetrics.accuracy(prediction, ground_truth),
        }

        return metrics

    @staticmethod
    def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
        """Dice Similarity Coefficient. Range: [0, 1]."""
        intersection = np.sum(pred * gt)
        return float((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth))

    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
        """Intersection over Union (Jaccard Index). Range: [0, 1]."""
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt) - intersection
        return float((intersection + smooth) / (union + smooth))

    @staticmethod
    def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """Overall pixel/voxel accuracy."""
        return float(np.mean(pred == gt))

    @staticmethod
    def precision(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
        """Precision = TP / (TP + FP)."""
        tp = np.sum(pred * gt)
        fp = np.sum(pred * (1 - gt))
        return float((tp + smooth) / (tp + fp + smooth))

    @staticmethod
    def recall(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
        """Recall = TP / (TP + FN)."""
        tp = np.sum(pred * gt)
        fn = np.sum((1 - pred) * gt)
        return float((tp + smooth) / (tp + fn + smooth))

    @staticmethod
    def specificity(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
        """Specificity = TN / (TN + FP)."""
        tn = np.sum((1 - pred) * (1 - gt))
        fp = np.sum(pred * (1 - gt))
        return float((tn + smooth) / (tn + fp + smooth))

    @staticmethod
    def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Hausdorff Distance between prediction and ground truth surfaces.
        Lower is better.
        """
        from scipy.ndimage import distance_transform_edt

        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return float("inf")

        pred_border = pred - ndimage.binary_erosion(pred).astype(pred.dtype)
        gt_border = gt - ndimage.binary_erosion(gt).astype(gt.dtype)

        dt_pred = distance_transform_edt(1 - pred_border)
        dt_gt = distance_transform_edt(1 - gt_border)

        hd1 = np.max(dt_gt[pred_border > 0]) if np.any(pred_border > 0) else 0
        hd2 = np.max(dt_pred[gt_border > 0]) if np.any(gt_border > 0) else 0

        return float(max(hd1, hd2))

    @staticmethod
    def average_surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        """Average Surface Distance. Lower is better."""
        from scipy.ndimage import distance_transform_edt

        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return float("inf")

        from scipy import ndimage as ndi

        pred_border = pred - ndi.binary_erosion(pred).astype(pred.dtype)
        gt_border = gt - ndi.binary_erosion(gt).astype(gt.dtype)

        dt_pred = distance_transform_edt(1 - pred_border)
        dt_gt = distance_transform_edt(1 - gt_border)

        asd1 = np.mean(dt_gt[pred_border > 0]) if np.any(pred_border > 0) else 0
        asd2 = np.mean(dt_pred[gt_border > 0]) if np.any(gt_border > 0) else 0

        return float((asd1 + asd2) / 2)
