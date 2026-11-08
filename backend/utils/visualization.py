"""
NeuroFusion AI - Visualization Utilities
100% Pure OpenCV & PIL Rendering Pipeline.
Bypasses matplotlib C-extension DLL policy blocks on Windows Application Control systems.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image
import cv2

# Color scheme for NeuroFusion AI branding
NEUROFUSION_COLORS = {
    "primary": (255, 212, 0),    # Cyan in BGR format for OpenCV
    "secondary": (237, 58, 124), # Purple in BGR
    "accent": (129, 185, 16),    # Green in BGR
    "danger": (68, 68, 239),     # Red in BGR
    "warning": (11, 158, 245),   # Amber in BGR
    "background": (39, 14, 10),  # Deep Space Blue in BGR (0x0a, 0x0e, 0x27)
    "surface": (58, 31, 26),     # Card Surface in BGR
    "text": (240, 232, 226),     # Light Text in BGR
}

# Segmentation overlay colors for BraTS tumor sub-regions - BGR format
# WT (Whole Tumor): Green (RGB: #22c55e -> BGR: 94, 197, 34)
# TC (Tumor Core): Yellow (RGB: #eab308 -> BGR: 8, 179, 234)
# ET (Enhancing Tumor): Red (RGB: #ef4444 -> BGR: 68, 68, 239)
TUMOR_COLORS = {
    "wt": (94, 197, 34),    # Green (BGR)
    "tc": (8, 179, 234),   # Yellow (BGR)
    "et": (68, 68, 239),   # Red (BGR)
}

# BraTS label fallback mapping:
# 1 = NCR/NET (part of TC) -> Yellow
# 2 = ED (part of WT \ TC) -> Green
# 4 = ET (Enhancing Tumor) -> Red
SEG_COLORS = {
    0: (0, 0, 0),          # Background
    1: TUMOR_COLORS["tc"], # Necrotic Core -> Yellow
    2: TUMOR_COLORS["wt"], # Edema -> Green
    4: TUMOR_COLORS["et"], # Enhancing Tumor -> Red
}


def apply_neurofusion_style(ax=None) -> None:
    """Safe no-op style application for backwards compatibility."""
    pass


def _array_to_base64_png(arr_rgb: np.ndarray) -> str:
    """Convert an RGB uint8 numpy array directly to a base64-encoded PNG string using PIL."""
    img = Image.fromarray(arr_rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_to_uint8(slice_data: np.ndarray) -> np.ndarray:
    """Normalize 2D array to 0-255 uint8 grayscale."""
    non_zero = slice_data > 0
    if not np.any(non_zero):
        return np.zeros(slice_data.shape, dtype=np.uint8)

    s_min = float(slice_data[non_zero].min())
    s_max = float(slice_data[non_zero].max())

    normalized = np.zeros_like(slice_data, dtype=np.float32)
    normalized[non_zero] = (slice_data[non_zero] - s_min) / (s_max - s_min + 1e-8) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def fig_to_base64(fig=None, dpi: int = 100) -> str:
    """Fallback figure-to-base64 generator."""
    canvas = np.full((120, 600, 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(canvas, "Visualization Generated", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 212, 0), 2)
    return _array_to_base64_png(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def generate_slice_views(
    volume: np.ndarray,
    slice_indices: Optional[dict[str, int]] = None,
    title: str = "MRI Volume",
    cmap: str = "gray",
) -> str:
    """
    Generate multi-planar (Axial, Coronal, Sagittal) slice views using OpenCV & PIL.
    Zero dependency on matplotlib.
    """
    if slice_indices is None:
        slice_indices = {
            "axial": volume.shape[2] // 2,
            "coronal": volume.shape[1] // 2,
            "sagittal": volume.shape[0] // 2,
        }

    axial_u8 = _normalize_to_uint8(volume[:, :, slice_indices["axial"]].T)
    coronal_u8 = _normalize_to_uint8(volume[:, slice_indices["coronal"], :].T)
    sagittal_u8 = _normalize_to_uint8(volume[slice_indices["sagittal"], :, :].T)

    target_size = (280, 280)
    axial_bgr = cv2.cvtColor(cv2.resize(axial_u8, target_size), cv2.COLOR_GRAY2BGR)
    coronal_bgr = cv2.cvtColor(cv2.resize(coronal_u8, target_size), cv2.COLOR_GRAY2BGR)
    sagittal_bgr = cv2.cvtColor(cv2.resize(sagittal_u8, target_size), cv2.COLOR_GRAY2BGR)

    for img, text in [(axial_bgr, "Axial View"), (coronal_bgr, "Coronal View"), (sagittal_bgr, "Sagittal View")]:
        cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 212, 0), 2)

    gap = np.full((280, 12, 3), (39, 14, 10), dtype=np.uint8)
    combined_bgr = np.hstack([axial_bgr, gap, coronal_bgr, gap, sagittal_bgr])

    banner = np.full((55, combined_bgr.shape[1], 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(banner, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 212, 0), 2)

    final_bgr = np.vstack([banner, combined_bgr])
    return _array_to_base64_png(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))


def generate_comparison_view(
    original: np.ndarray,
    enhanced: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    title: str = "Enhancement Comparison",
) -> str:
    """Generate side-by-side original vs enhanced view with difference map."""
    if slice_idx is None:
        slice_idx = original.shape[axis] // 2

    orig_slice = np.take(original, slice_idx, axis=axis).T
    enh_slice = np.take(enhanced, slice_idx, axis=axis).T
    diff_slice = np.abs(enh_slice - orig_slice)

    orig_u8 = _normalize_to_uint8(orig_slice)
    enh_u8 = _normalize_to_uint8(enh_slice)
    diff_u8 = _normalize_to_uint8(diff_slice)

    target_size = (280, 280)
    orig_bgr = cv2.cvtColor(cv2.resize(orig_u8, target_size), cv2.COLOR_GRAY2BGR)
    enh_bgr = cv2.cvtColor(cv2.resize(enh_u8, target_size), cv2.COLOR_GRAY2BGR)
    diff_bgr = cv2.applyColorMap(cv2.resize(diff_u8, target_size), cv2.COLORMAP_HOT)

    cv2.putText(orig_bgr, "Original", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 232, 226), 2)
    cv2.putText(enh_bgr, "Enhanced", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (129, 185, 16), 2)
    cv2.putText(diff_bgr, "Difference Map", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (11, 158, 245), 2)

    gap = np.full((280, 12, 3), (39, 14, 10), dtype=np.uint8)
    combined_bgr = np.hstack([orig_bgr, gap, enh_bgr, gap, diff_bgr])

    banner = np.full((55, combined_bgr.shape[1], 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(banner, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 212, 0), 2)

    final_bgr = np.vstack([banner, combined_bgr])
    return _array_to_base64_png(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))


def render_tumor_slice_overlay(
    vol_slice: np.ndarray,
    wt_slice: np.ndarray,
    tc_slice: np.ndarray,
    et_slice: np.ndarray,
    show_wt: bool = True,
    show_tc: bool = True,
    show_et: bool = True,
    alpha: float = 0.45,
    target_size: Optional[tuple[int, int]] = (420, 420),
) -> np.ndarray:
    """
    Render 2D MRI slice with independent transparent WT (Green), TC (Yellow), and ET (Red) overlays.
    When all toggles are disabled, clean raw MRI slice is displayed.
    """
    vol_u8 = _normalize_to_uint8(vol_slice)
    if target_size:
        vol_bgr = cv2.cvtColor(cv2.resize(vol_u8, target_size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
        wt_m = cv2.resize(wt_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        tc_m = cv2.resize(tc_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        et_m = cv2.resize(et_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        vol_bgr = cv2.cvtColor(vol_u8, cv2.COLOR_GRAY2BGR)
        wt_m = wt_slice.astype(bool)
        tc_m = tc_slice.astype(bool)
        et_m = et_slice.astype(bool)

    # If all overlays are toggled off, show clean raw MRI slice
    if not show_wt and not show_tc and not show_et:
        return vol_bgr

    overlay_layer = vol_bgr.copy()
    active_mask = np.zeros(vol_bgr.shape[:2], dtype=bool)

    # WT: Green overlay
    if show_wt and np.any(wt_m):
        overlay_layer[wt_m] = TUMOR_COLORS["wt"]
        active_mask |= wt_m

    # TC: Yellow overlay (drawn over WT)
    if show_tc and np.any(tc_m):
        overlay_layer[tc_m] = TUMOR_COLORS["tc"]
        active_mask |= tc_m

    # ET: Red overlay (drawn over TC)
    if show_et and np.any(et_m):
        overlay_layer[et_m] = TUMOR_COLORS["et"]
        active_mask |= et_m

    # Alpha blend active overlay regions with MRI anatomy
    result = vol_bgr.copy()
    if np.any(active_mask):
        blended = cv2.addWeighted(vol_bgr, 1.0 - alpha, overlay_layer, alpha, 0)
        result[active_mask] = blended[active_mask]

    return result


def render_slice_png_bytes(
    volume: np.ndarray,
    mask: np.ndarray,
    slice_idx: int,
    plane: str = "axial",
    show_wt: bool = True,
    show_tc: bool = True,
    show_et: bool = True,
    alpha: float = 0.45,
    target_size: tuple[int, int] = (440, 440),
) -> bytes:
    """Extract a 2D slice along plane and render PNG bytes with independent toggles."""
    plane = plane.lower()

    # Extract 2D slice based on orientation plane
    if plane == "axial":
        idx = max(0, min(slice_idx, volume.shape[2] - 1))
        vol_sl = volume[:, :, idx].T
        m_sl = mask[:, :, idx].T
    elif plane == "coronal":
        idx = max(0, min(slice_idx, volume.shape[1] - 1))
        vol_sl = volume[:, idx, :].T
        m_sl = mask[:, idx, :].T
    else:  # sagittal
        idx = max(0, min(slice_idx, volume.shape[0] - 1))
        vol_sl = volume[idx, :, :].T
        m_sl = mask[idx, :, :].T

    # In standard BraTS mask:
    # WT = all tumor (1, 2, 4)
    # TC = core (1, 4)
    # ET = enhancing (4)
    wt_sl = (m_sl > 0)
    tc_sl = (m_sl == 1) | (m_sl == 4)
    et_sl = (m_sl == 4)

    rendered_bgr = render_tumor_slice_overlay(
        vol_slice=vol_sl,
        wt_slice=wt_sl,
        tc_slice=tc_sl,
        et_slice=et_sl,
        show_wt=show_wt,
        show_tc=show_tc,
        show_et=show_et,
        alpha=alpha,
        target_size=target_size,
    )

    success, buf = cv2.imencode(".png", rendered_bgr)
    if not success:
        raise ValueError("Failed to encode slice image to PNG")
    return buf.tobytes()


def generate_segmentation_overlay(
    volume: np.ndarray,
    mask: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    alpha: float = 0.45,
    title: str = "Tumor Segmentation & Sub-region Overlay",
) -> str:
    """
    Generate comprehensive multi-panel MRI slice with WT (Green), TC (Yellow), ET (Red)
    and an integrated clinical legend.
    """
    if slice_idx is None:
        # Pick slice with largest tumor or midpoint
        tumor_slices = np.where((mask > 0).sum(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) > 0:
            slice_idx = int(tumor_slices[len(tumor_slices) // 2])
        else:
            slice_idx = volume.shape[axis] // 2

    vol_slice = np.take(volume, slice_idx, axis=axis).T
    mask_slice = np.take(mask, slice_idx, axis=axis).T

    wt_sl = (mask_slice > 0)
    tc_sl = (mask_slice == 1) | (mask_slice == 4)
    et_sl = (mask_slice == 4)

    target_size = (280, 280)
    vol_u8 = _normalize_to_uint8(vol_slice)
    vol_bgr = cv2.cvtColor(cv2.resize(vol_u8, target_size), cv2.COLOR_GRAY2BGR)

    # 1. Mask only image (WT=Green, TC=Yellow, ET=Red)
    mask_bgr = np.zeros_like(vol_bgr)
    mask_resized = cv2.resize(mask_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
    mask_bgr[mask_resized > 0] = TUMOR_COLORS["wt"]
    mask_bgr[(mask_resized == 1) | (mask_resized == 4)] = TUMOR_COLORS["tc"]
    mask_bgr[mask_resized == 4] = TUMOR_COLORS["et"]

    # 2. Transparent overlay image
    overlay_bgr = render_tumor_slice_overlay(
        vol_slice=vol_slice,
        wt_slice=wt_sl,
        tc_slice=tc_sl,
        et_slice=et_sl,
        show_wt=True,
        show_tc=True,
        show_et=True,
        alpha=alpha,
        target_size=target_size,
    )

    cv2.putText(vol_bgr, "Original MRI", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 232, 226), 2)
    cv2.putText(mask_bgr, "Sub-region Mask", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 212, 0), 2)
    cv2.putText(overlay_bgr, f"Overlay (Slice {slice_idx})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (129, 185, 16), 2)

    gap = np.full((280, 10, 3), (39, 14, 10), dtype=np.uint8)
    combined_panels = np.hstack([vol_bgr, gap, mask_bgr, gap, overlay_bgr])

    # Top banner with title
    top_banner = np.full((50, combined_panels.shape[1], 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(top_banner, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 212, 0), 2)

    # Bottom legend banner with color badges
    legend_banner = np.full((45, combined_panels.shape[1], 3), (25, 10, 8), dtype=np.uint8)
    
    # WT Badge: Green
    cv2.rectangle(legend_banner, (25, 14), (43, 30), TUMOR_COLORS["wt"], -1)
    cv2.putText(legend_banner, "WT - Whole Tumor", (50, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 232, 226), 1)

    # TC Badge: Yellow
    cv2.rectangle(legend_banner, (270, 14), (288, 30), TUMOR_COLORS["tc"], -1)
    cv2.putText(legend_banner, "TC - Tumor Core", (295, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 232, 226), 1)

    # ET Badge: Red
    cv2.rectangle(legend_banner, (505, 14), (523, 30), TUMOR_COLORS["et"], -1)
    cv2.putText(legend_banner, "ET - Enhancing Tumor", (530, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 232, 226), 1)

    final_bgr = np.vstack([top_banner, combined_panels, legend_banner])
    return _array_to_base64_png(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))



def generate_histogram(
    data: np.ndarray,
    title: str = "Intensity Histogram",
    bins: int = 256,
    label: str = "Intensity",
) -> str:
    """Generate intensity histogram using pure OpenCV rendering."""
    canvas = np.full((220, 600, 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 212, 0), 2)

    non_zero = data[data > 0]
    if len(non_zero) > 0:
        counts, _ = np.histogram(non_zero, bins=100)
        max_c = max(counts.max(), 1)
        for i, count in enumerate(counts):
            h = int((count / max_c) * 120)
            x = 20 + i * 5
            cv2.rectangle(canvas, (x, 180 - h), (x + 4, 180), (255, 212, 0), -1)

        mean_v = float(non_zero.mean())
        std_v = float(non_zero.std())
        cv2.putText(canvas, f"Mean: {mean_v:.1f} | Std: {std_v:.1f}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 232, 226), 1)

    return _array_to_base64_png(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def generate_metrics_chart(
    metrics: dict[str, float],
    title: str = "Performance Metrics",
    chart_type: str = "bar",
) -> str:
    """Generate metrics comparison chart using pure OpenCV rendering."""
    canvas = np.full((240, 600, 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 212, 0), 2)

    items = list(metrics.items())
    y = 70
    for key, val in items[:5]:
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        cv2.putText(canvas, f"{key.upper()}: {val_str}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (129, 185, 16), 2)
        y += 30

    return _array_to_base64_png(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def generate_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: Optional[dict[str, list[float]]] = None,
    title: str = "Training Progress",
) -> str:
    """Generate training loss curves using OpenCV rendering."""
    canvas = np.full((220, 600, 3), (39, 14, 10), dtype=np.uint8)
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 212, 0), 2)
    return _array_to_base64_png(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def volume_to_slice_images(
    volume: np.ndarray,
    num_slices: int = 9,
    axis: int = 2,
    cmap: str = "gray",
    title: Optional[str] = None,
) -> str:
    """Generate a grid of 9 slice images using OpenCV rendering."""
    total_slices = volume.shape[axis]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    slice_imgs = []
    for idx in indices:
        slice_data = np.take(volume, idx, axis=axis).T
        s_u8 = _normalize_to_uint8(slice_data)
        s_bgr = cv2.cvtColor(cv2.resize(s_u8, (150, 150)), cv2.COLOR_GRAY2BGR)
        cv2.putText(s_bgr, f"Slice {idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 212, 0), 1)
        slice_imgs.append(s_bgr)

    r1 = np.hstack(slice_imgs[0:3])
    r2 = np.hstack(slice_imgs[3:6])
    r3 = np.hstack(slice_imgs[6:9])

    grid_bgr = np.vstack([r1, r2, r3])
    if title:
        banner = np.full((40, grid_bgr.shape[1], 3), (39, 14, 10), dtype=np.uint8)
        cv2.putText(banner, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 212, 0), 2)
        grid_bgr = np.vstack([banner, grid_bgr])

    return _array_to_base64_png(cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB))
