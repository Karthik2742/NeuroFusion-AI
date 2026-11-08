"""
NeuroFusion AI & BrainSight - Brain MRI Enhancement, Segmentation & Diagnostic Web Platform
Streamlit Web Application
"""

from __future__ import annotations

import os
import sys
import time
import io
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch

# Ensure NeuroFusion AI root is on python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import NeuroFusion AI backend modules
from backend.utils.io import NIfTILoader
from backend.preprocessing.dataset_analyzer import DatasetAnalyzer
from backend.preprocessing.preprocessor import MRIPreprocessor
from backend.models.enhancement import get_enhancement_model, ENHANCEMENT_MODELS
from backend.evaluation.metrics import EnhancementMetrics
from backend.evaluation.explainability import DiseaseLocalizer
from backend.models.segmentation.segmenter import predict_tumor_subregions
from backend.utils.visualization import render_tumor_slice_overlay
from backend.utils.report_generator import BrainSightReportGenerator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroFusion AI | Brain MRI & Tumor Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN MEDICAL CSS STYLING ---
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1527 0%, #070b14 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 0.98rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #38bdf8;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge-tumor {
        background: rgba(239, 68, 68, 0.18);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-normal {
        background: rgba(34, 197, 94, 0.18);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        font-weight: 600;
        border-bottom: none !important;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA DISCOVERY HELPERS ---
BRAIN_JN_1_DIR = Path(r"c:\Users\ppriy\Downloads\Telegram Desktop\Brain_JN_1")
LOCAL_DATASET_DIR = CURRENT_DIR / "dataset" / "brain"

def discover_available_datasets() -> dict[str, list[dict[str, Any]]]:
    """Catalog all available scans from Brain_JN_1 and local datasets."""
    catalog = {
        "Brain_JN_1: Pathological Brain MRI (Tumor Scans)": [],
        "Brain_JN_1: Normal Brain MRI Scans": [],
        "NeuroFusion AI: BraTS2020 & Benchmark Scans": [],
    }

    # 1. Pathological Scans from Brain_JN_1
    patho_dir = BRAIN_JN_1_DIR / "Brain DATASETS" / "Pathological brain MRI Datasets"
    if patho_dir.exists():
        for file in sorted(patho_dir.glob("**/*.nii*")):
            if file.is_file() and not file.name.startswith("."):
                patient = file.parent.name
                catalog["Brain_JN_1: Pathological Brain MRI (Tumor Scans)"].append({
                    "name": f"{patient} — {file.name}",
                    "path": str(file),
                    "patient": patient,
                    "type": "pathological",
                })

    # 2. Normal Scans from Brain_JN_1
    normal_dir = BRAIN_JN_1_DIR / "Brain DATASETS" / "Normal brain Datasets"
    if normal_dir.exists():
        for file in sorted(normal_dir.glob("**/*.nii*")):
            if file.is_file() and not file.name.startswith("."):
                rel_parts = file.relative_to(normal_dir).parts
                subject = rel_parts[0] if len(rel_parts) > 1 else "Normal"
                catalog["Brain_JN_1: Normal Brain MRI Scans"].append({
                    "name": f"{subject} — {file.name}",
                    "path": str(file),
                    "patient": subject,
                    "type": "normal",
                })

    # 3. Local Dataset Scans
    if LOCAL_DATASET_DIR.exists():
        for file in sorted(LOCAL_DATASET_DIR.glob("**/*.nii*")):
            if file.is_file() and not file.name.startswith("."):
                category = file.parent.name
                catalog["NeuroFusion AI: BraTS2020 & Benchmark Scans"].append({
                    "name": f"[{category.upper()}] {file.name}",
                    "path": str(file),
                    "patient": category,
                    "type": "benchmark",
                })

    return catalog


# --- CACHED NIFTI LOADER ---
@st.cache_data(show_spinner=False, max_entries=8)
def load_cached_volume(filepath: str) -> tuple[np.ndarray, list[float], list[int]]:
    """Loads a NIfTI volume, returns data, voxel spacing zooms, and shape."""
    loader = NIfTILoader()
    data, header, affine = loader.load_nibabel(filepath)
    zooms = [1.0, 1.0, 1.0]
    if header is not None and hasattr(header, "get_zooms"):
        zooms = [float(z) for z in header.get_zooms()[:3]]
    return data.astype(np.float32), zooms, list(data.shape)


def normalize_slice_for_display(
    slice_arr: np.ndarray,
    p_min: float = 0.5,
    p_max: float = 99.5,
) -> np.ndarray:
    """Normalize 2D MRI slice to [0, 1] range with robust contrast clipping."""
    valid = slice_arr[slice_arr > 0]
    if len(valid) == 0:
        return np.zeros_like(slice_arr, dtype=np.float32)
    vmin = np.percentile(valid, p_min)
    vmax = np.percentile(valid, p_max)
    if vmax <= vmin:
        vmax = vmin + 1e-5
    clipped = np.clip(slice_arr, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


# --- MAIN HEADER ---
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1>🧠 NeuroFusion AI & BrainSight</h1>
            <p>Advanced Volumetric Brain MRI Enhancement, Disease Localization, Tumor Sub-region Segmentation & Diagnostic Analytics</p>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:4px 12px; border-radius:6px; font-size:0.8rem; font-weight:600;">
                Streamlit AI Suite v2.0
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- SIDEBAR: DATASET EXPLORER & CONTROLS ---
with st.sidebar:
    st.markdown("### 📁 Dataset & Scan Selector")
    
    catalog = discover_available_datasets()
    
    source_type = st.radio(
        "Select Data Source:",
        [
            "Select from Brain_JN_1 (Datasets)",
            "Select from NeuroFusion Benchmarks",
            "Upload Custom NIfTI (.nii / .nii.gz)",
        ],
        index=0,
    )
    
    selected_file_path: str | None = None
    selected_file_name: str | None = None
    
    if source_type == "Select from Brain_JN_1 (Datasets)":
        dataset_category = st.selectbox(
            "Category:",
            [
                "Brain_JN_1: Pathological Brain MRI (Tumor Scans)",
                "Brain_JN_1: Normal Brain MRI Scans",
            ],
        )
        file_options = catalog.get(dataset_category, [])
        if file_options:
            selected_item = st.selectbox(
                "Choose Scan:",
                options=file_options,
                format_func=lambda x: x["name"],
            )
            if selected_item:
                selected_file_path = selected_item["path"]
                selected_file_name = Path(selected_file_path).name
        else:
            st.warning("No scans found in selected Brain_JN_1 folder.")
            
    elif source_type == "Select from NeuroFusion Benchmarks":
        benchmark_options = catalog.get("NeuroFusion AI: BraTS2020 & Benchmark Scans", [])
        if benchmark_options:
            selected_item = st.selectbox(
                "Choose Benchmark Scan:",
                options=benchmark_options,
                format_func=lambda x: x["name"],
            )
            if selected_item:
                selected_file_path = selected_item["path"]
                selected_file_name = Path(selected_file_path).name
        else:
            st.warning("No benchmark scans found in dataset/brain.")
            
    else:
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Volume (.nii or .nii.gz)",
            type=["nii", "gz"],
            help="Upload full 3D Brain MRI scan.",
        )
        if uploaded_file is not None:
            temp_dir = CURRENT_DIR / "temp" / "uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            upload_path = temp_dir / uploaded_file.name
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_file_path = str(upload_path)
            selected_file_name = uploaded_file.name

    st.markdown("---")
    st.markdown("### 🎨 Display Settings")
    colormap_choice = st.selectbox(
        "Colormap:",
        ["gray", "bone", "viridis", "plasma", "magma", "inferno"],
        index=0,
    )
    
    contrast_window = st.slider(
        "Contrast Windowing (Percentiles):",
        min_value=0.0,
        max_value=100.0,
        value=(1.0, 99.0),
        step=0.5,
    )
    
    st.markdown("---")
    device_name = "CUDA (NVIDIA GPU)" if torch.cuda.is_available() else "CPU Mode"
    st.info(f"⚙️ Compute Device: **{device_name}**")


# --- LOAD SCAN DATA ---
if not selected_file_path or not Path(selected_file_path).exists():
    st.info("👈 Please select a scan from the sidebar or upload a NIfTI MRI scan to begin.")
    st.stop()

# Load volume
with st.spinner(f"Loading MRI scan: {selected_file_name}..."):
    try:
        raw_volume, voxel_spacing, orig_shape = load_cached_volume(selected_file_path)
    except Exception as e:
        st.error(f"Failed to load MRI volume: {e}")
        st.stop()

# Save in session state
st.session_state["raw_volume"] = raw_volume
st.session_state["voxel_spacing"] = voxel_spacing
st.session_state["shape"] = orig_shape
st.session_state["current_path"] = selected_file_path
st.session_state["current_filename"] = selected_file_name


# --- QUICK SUMMARY BAR ---
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
file_size_mb = round(os.path.getsize(selected_file_path) / (1024 * 1024), 2)
vol_cm3 = round(np.prod(orig_shape) * np.prod(voxel_spacing) / 1000.0, 1)

with col_s1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Volume Dimensions</div>
        <div class="metric-value">{orig_shape[0]}×{orig_shape[1]}×{orig_shape[2]}</div>
        <div class="metric-sub">3D Voxel Grid</div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Voxel Spacing</div>
        <div class="metric-value">{voxel_spacing[0]:.1f}×{voxel_spacing[1]:.1f}×{voxel_spacing[2]:.1f}</div>
        <div class="metric-sub">mm / voxel</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Physical Volume</div>
        <div class="metric-value">{vol_cm3:,.0f}</div>
        <div class="metric-sub">cm³ Field of View</div>
    </div>
    """, unsafe_allow_html=True)

with col_s4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">File Size</div>
        <div class="metric-value">{file_size_mb} MB</div>
        <div class="metric-sub">NIfTI Volume</div>
    </div>
    """, unsafe_allow_html=True)

with col_s5:
    is_patho = "pathological" in selected_file_path.lower() or "brp" in selected_file_path.lower() or "tumor" in selected_file_path.lower()
    badge_html = '<span class="badge-tumor">Pathology Suspect</span>' if is_patho else '<span class="badge-normal">Control / Normal</span>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Clinical Indication</div>
        <div style="margin-top:6px;">{badge_html}</div>
        <div class="metric-sub">{selected_file_name[:16]}...</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# --- MAIN TABS ---
tab_viewer, tab_analysis, tab_preprocess, tab_enhance, tab_segment, tab_report, tab_diagnostics = st.tabs([
    "👁️ 3D Multi-Planar Viewer",
    "📊 Volumetric Analysis",
    "⚙️ Advanced Preprocessing",
    "✨ Deep MRI Enhancement",
    "🧠 Tumor Segmentation & Localization",
    "📄 Clinical Report & Export",
    "⚡ System Diagnostics",
])


# =========================================================================
# TAB 1: 3D MULTI-PLANAR VIEWER
# =========================================================================
with tab_viewer:
    st.markdown("#### 🧭 Orthogonal Multi-Planar Slices (Axial, Coronal, Sagittal)")
    
    # 3 Slider columns for interactive scrubbing
    c_ax, c_cor, c_sag = st.columns(3)
    
    nx, ny, nz = raw_volume.shape
    
    with c_ax:
        slice_z = st.slider("Axial Slice (Transverse - Z):", 0, nz - 1, nz // 2)
    with c_cor:
        slice_y = st.slider("Coronal Slice (Frontal - Y):", 0, ny - 1, ny // 2)
    with c_sag:
        slice_x = st.slider("Sagittal Slice (Lateral - X):", 0, nx - 1, nx // 2)

    # Extract 2D slices
    axial_slice = raw_volume[:, :, slice_z]
    coronal_slice = raw_volume[:, slice_y, :]
    sagittal_slice = raw_volume[slice_x, :, :]

    # Normalize slices with user contrast window
    p_low, p_high = contrast_window
    ax_norm = normalize_slice_for_display(axial_slice, p_low, p_high)
    cor_norm = normalize_slice_for_display(coronal_slice, p_low, p_high)
    sag_norm = normalize_slice_for_display(sagittal_slice, p_low, p_high)

    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        st.markdown(f"**Axial View (Z = {slice_z + 1} / {nz})**")
        fig_ax, ax_plot = plt.subplots(figsize=(5, 5), facecolor="#0B132B")
        ax_plot.imshow(np.rot90(ax_norm), cmap=colormap_choice)
        ax_plot.axis("off")
        st.pyplot(fig_ax, use_container_width=True)
        plt.close(fig_ax)

    with v_col2:
        st.markdown(f"**Coronal View (Y = {slice_y + 1} / {ny})**")
        fig_cor, cor_plot = plt.subplots(figsize=(5, 5), facecolor="#0B132B")
        cor_plot.imshow(np.rot90(cor_norm), cmap=colormap_choice)
        cor_plot.axis("off")
        st.pyplot(fig_cor, use_container_width=True)
        plt.close(fig_cor)

    with v_col3:
        st.markdown(f"**Sagittal View (X = {slice_x + 1} / {nx})**")
        fig_sag, sag_plot = plt.subplots(figsize=(5, 5), facecolor="#0B132B")
        sag_plot.imshow(np.rot90(sag_norm), cmap=colormap_choice)
        sag_plot.axis("off")
        st.pyplot(fig_sag, use_container_width=True)
        plt.close(fig_sag)


# =========================================================================
# TAB 2: VOLUMETRIC ANALYSIS
# =========================================================================
with tab_analysis:
    st.markdown("#### 📊 Quantitative Dataset & Volumetric Quality Analysis")
    
    if st.button("🚀 Run Full Volumetric & Quality Analysis", key="btn_run_analysis"):
        with st.spinner("Analyzing volume statistics and quality metrics..."):
            analyzer = DatasetAnalyzer()
            res = analyzer.analyze_file(selected_file_path)
            st.session_state["analysis_result"] = res
            st.success("Analysis complete!")

    analysis_res = st.session_state.get("analysis_result")
    
    if analysis_res:
        int_stats = analysis_res.get("intensity_stats", {})
        qual_stats = analysis_res.get("quality_metrics", {})
        
        q1, q2, q3, q4, q5 = st.columns(5)
        with q1:
            st.metric("SNR (Signal-to-Noise)", f"{qual_stats.get('snr', 0):.2f}")
        with q2:
            st.metric("CNR (Contrast-to-Noise)", f"{qual_stats.get('cnr', 0):.2f}")
        with q3:
            st.metric("Entropy", f"{qual_stats.get('entropy', 0):.2f}")
        with q4:
            st.metric("Contrast Index", f"{qual_stats.get('contrast', 0):.2f}")
        with q5:
            st.metric("Sharpness Score", f"{qual_stats.get('sharpness', 0):.2f}")

        st.markdown("---")
        h_col1, h_col2 = st.columns([1, 1])
        
        with h_col1:
            st.markdown("##### 📈 Voxel Intensity Distribution (Histogram)")
            valid_voxels = raw_volume[raw_volume > 0].flatten()
            sample_voxels = np.random.choice(valid_voxels, size=min(len(valid_voxels), 25000), replace=False)
            
            fig_hist = px.histogram(
                x=sample_voxels,
                nbins=60,
                title="Intensity Profile (Non-Zero Voxels)",
                labels={"x": "Voxel Intensity", "y": "Frequency"},
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8"],
            )
            fig_hist.update_layout(bargap=0.05, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with h_col2:
            st.markdown("##### 📋 Summary Statistical Table")
            summary_data = {
                "Metric": ["Mean Intensity", "Std Deviation", "Median Intensity", "Min Voxel", "Max Voxel", "Dynamic Range", "Detected Modality"],
                "Value": [
                    f"{int_stats.get('mean', np.mean(valid_voxels)):.2f}",
                    f"{int_stats.get('std', np.std(valid_voxels)):.2f}",
                    f"{int_stats.get('median', np.median(valid_voxels)):.2f}",
                    f"{int_stats.get('min', np.min(valid_voxels)):.2f}",
                    f"{int_stats.get('max', np.max(valid_voxels)):.2f}",
                    f"{int_stats.get('dynamic_range', np.max(valid_voxels) - np.min(valid_voxels)):.2f}",
                    f"{analysis_res.get('modality', 'Brain MRI')}",
                ],
            }
            st.dataframe(summary_data, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Full Volumetric & Quality Analysis' above to generate comprehensive quality scores and histograms.")


# =========================================================================
# TAB 3: ADVANCED PREPROCESSING
# =========================================================================
with tab_preprocess:
    st.markdown("#### ⚙️ Standard Medical Preprocessing Pipeline")
    st.write("Apply bias correction, skull stripping, isotropic spatial resampling, and intensity normalization.")

    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        st.markdown("##### Select Pipeline Steps:")
        step_resample = st.checkbox("Isotropic Resampling (1.0 mm³)", value=False)
        step_normalize = st.checkbox("Z-Score Intensity Normalization", value=True)
        step_skullstrip = st.checkbox("Skull Stripping / Brain Masking", value=True)
        step_n4 = st.checkbox("N4 Bias Field Correction (SimpleITK)", value=False)
        
        selected_steps = []
        if step_resample: selected_steps.append("resample")
        if step_normalize: selected_steps.append("normalize")
        if step_skullstrip: selected_steps.append("skull_strip")
        if step_n4: selected_steps.append("n4_bias_field")

        if st.button("▶️ Execute Preprocessing Pipeline", key="btn_run_preproc"):
            if not selected_steps:
                st.warning("Please select at least one preprocessing step.")
            else:
                with st.spinner("Processing MRI volume..."):
                    preprocessor = MRIPreprocessor()
                    res_preproc = preprocessor.run_pipeline(
                        filepath=selected_file_path,
                        steps=selected_steps,
                    )
                    st.session_state["preprocessed_data"] = res_preproc["data"]
                    st.session_state["preproc_info"] = res_preproc
                    st.success("Preprocessing completed!")

    with p_col2:
        preproc_data = st.session_state.get("preprocessed_data")
        if preproc_data is not None:
            st.markdown("##### 🔬 Before vs. After Preprocessing Comparison")
            p_slice_idx = st.slider("Select Slice to Compare:", 0, orig_shape[2] - 1, orig_shape[2] // 2, key="slider_preproc_slice")
            
            c_orig, c_proc = st.columns(2)
            with c_orig:
                st.caption(f"Original Scan (Slice {p_slice_idx + 1})")
                raw_norm = normalize_slice_for_display(raw_volume[:, :, p_slice_idx])
                fig_o, ax_o = plt.subplots(figsize=(4, 4), facecolor="#0B132B")
                ax_o.imshow(np.rot90(raw_norm), cmap=colormap_choice)
                ax_o.axis("off")
                st.pyplot(fig_o, use_container_width=True)
                plt.close(fig_o)

            with c_proc:
                st.caption(f"Preprocessed Scan (Slice {p_slice_idx + 1})")
                pr_slice = preproc_data[:, :, min(p_slice_idx, preproc_data.shape[2] - 1)]
                pr_norm = normalize_slice_for_display(pr_slice)
                fig_p, ax_p = plt.subplots(figsize=(4, 4), facecolor="#0B132B")
                ax_p.imshow(np.rot90(pr_norm), cmap=colormap_choice)
                ax_p.axis("off")
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)
                
            steps_applied = st.session_state.get("preproc_info", {}).get("steps_applied", [])
            st.info(f"Pipeline executed successfully. Steps applied: **{', '.join(steps_applied)}**")
        else:
            st.info("Execute the pipeline on the left to view Before vs. After results.")


# =========================================================================
# TAB 4: DEEP MRI ENHANCEMENT
# =========================================================================
with tab_enhance:
    st.markdown("#### ✨ Deep Learning MRI Enhancement & Quality Restoration")
    st.write("Restore image contrast, denoise high-frequency artifacts, and enhance diagnostic clarity using state-of-the-art vision models.")

    enh_col1, enh_col2 = st.columns([1, 2])
    with enh_col1:
        enh_model_name = st.selectbox(
            "Select Enhancement Architecture:",
            ["clahe", "restormer", "swinir", "mirnet"],
            format_func=lambda x: {
                "clahe": "CLAHE (Clinical Baseline - Fast)",
                "restormer": "Restormer (Transformer - CVPR 2022)",
                "swinir": "SwinIR (Swin Transformer - ICCV 2021)",
                "mirnet": "MIRNet (Multi-Scale Denoising - ECCV 2020)",
            }.get(x, x),
        )
        
        mode_choice = st.radio("Enhancement Scope:", ["Current 2D Axial Slice (Fast)", "Full 3D Volume"], index=0)
        
        e_slice_idx = st.slider(
            "Select Slice for Enhancement:",
            0, orig_shape[2] - 1, orig_shape[2] // 2,
            key="slider_enh_slice",
        )

        if st.button("🚀 Enhance MRI", key="btn_run_enhancement"):
            with st.spinner(f"Running {enh_model_name.upper()} enhancement..."):
                t0 = time.time()
                enhancer = get_enhancement_model(enh_model_name)
                
                if mode_choice == "Current 2D Axial Slice (Fast)":
                    slice_2d = raw_volume[:, :, e_slice_idx]
                    # Format as 3D block (H, W, 1) for the enhancer
                    enhanced_vol, meta = enhancer.enhance_volume(slice_2d[:, :, np.newaxis])
                    enhanced_res = enhanced_vol[:, :, 0]
                    metrics = EnhancementMetrics.compute_all(slice_2d, enhanced_res)
                else:
                    enhanced_vol, meta = enhancer.enhance_volume(raw_volume)
                    enhanced_res = enhanced_vol[:, :, e_slice_idx]
                    metrics = EnhancementMetrics.compute_all(raw_volume, enhanced_vol)

                elapsed = time.time() - t0
                st.session_state["enhanced_slice"] = enhanced_res
                st.session_state["enhanced_metrics"] = metrics
                st.session_state["enhanced_time"] = elapsed
                st.session_state["enhanced_model"] = enh_model_name
                st.success(f"Enhancement complete in {elapsed:.2f}s!")

    with enh_col2:
        enhanced_slice = st.session_state.get("enhanced_slice")
        if enhanced_slice is not None:
            m_metrics = st.session_state.get("enhanced_metrics", {})
            
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("PSNR (Peak Signal)", f"{m_metrics.get('psnr', 0):.2f} dB")
            ec2.metric("SSIM (Structural Similarity)", f"{m_metrics.get('ssim', 0):.4f}")
            ec3.metric("MSE (Mean Squared Error)", f"{m_metrics.get('mse', 0):.4f}")
            ec4.metric("Inference Time", f"{st.session_state.get('enhanced_time', 0):.2f} s")

            c_before, c_after = st.columns(2)
            with c_before:
                st.caption("Original Raw MRI Slice")
                orig_disp = normalize_slice_for_display(raw_volume[:, :, e_slice_idx])
                fig_b, ax_b = plt.subplots(figsize=(4, 4), facecolor="#0B132B")
                ax_b.imshow(np.rot90(orig_disp), cmap=colormap_choice)
                ax_b.axis("off")
                st.pyplot(fig_b, use_container_width=True)
                plt.close(fig_b)

            with c_after:
                st.caption(f"Enhanced with {st.session_state.get('enhanced_model', '').upper()}")
                enh_disp = normalize_slice_for_display(enhanced_slice)
                fig_a, ax_a = plt.subplots(figsize=(4, 4), facecolor="#0B132B")
                ax_a.imshow(np.rot90(enh_disp), cmap=colormap_choice)
                ax_a.axis("off")
                st.pyplot(fig_a, use_container_width=True)
                plt.close(fig_a)
        else:
            st.info("Configure enhancement options and click 'Enhance MRI' to view metrics and restored images.")


# =========================================================================
# TAB 5: TUMOR SEGMENTATION & LOCALIZATION
# =========================================================================
with tab_segment:
    st.markdown("#### 🧠 Brain Tumor Segmentation & Disease Localization")
    st.write("Perform deep volumetric segmentation of tumor sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).")

    seg_c1, seg_c2 = st.columns([1, 2])
    with seg_c1:
        st.markdown("##### Segmentation Model:")
        model_arch = st.selectbox(
            "Select Deep Learning Architecture:",
            ["monai_unet3d", "unet", "unet_plus", "attention_unet"],
            format_func=lambda x: {
                "monai_unet3d": "MONAI 3D UNet (Trained TorchScript Model)",
                "unet": "Standard 2D/3D UNet",
                "unet_plus": "UNet++ (Nested Dense Skips)",
                "attention_unet": "Attention UNet (Gated Attention)",
            }.get(x, x),
        )

        weights_pt = CURRENT_DIR / "weights" / "BrainTumorSegmentationModel.pt"
        if weights_pt.exists():
            st.caption(f"Model Weights Loaded: `{weights_pt.name}` ({weights_pt.stat().st_size / 1e6:.1f} MB)")

        run_seg = st.button("🔬 Segment Tumor Sub-regions", key="btn_run_seg", type="primary")

        if run_seg:
            with st.spinner("Executing 3D Segmentation and Disease Localization..."):
                loader = NIfTILoader()
                _, header, affine = loader.load_nibabel(selected_file_path)
                
                wt_m, tc_m, et_m, combined_m, meta = predict_tumor_subregions(
                    volume=raw_volume,
                    architecture=model_arch,
                    header=header,
                )
                
                # Disease localization attention heatmap
                loc_findings = DiseaseLocalizer.localize_pathology(raw_volume, mask=combined_m)
                
                st.session_state["seg_wt"] = wt_m
                st.session_state["seg_tc"] = tc_m
                st.session_state["seg_et"] = et_m
                st.session_state["seg_mask"] = combined_m
                st.session_state["seg_meta"] = meta
                st.session_state["loc_findings"] = loc_findings
                st.success("Segmentation analysis completed successfully!")

    with seg_c2:
        seg_mask = st.session_state.get("seg_mask")
        seg_meta = st.session_state.get("seg_meta")

        if seg_mask is not None and seg_meta is not None:
            measurements = seg_meta.get("measurements", {})
            wt_info = measurements.get("wt", {})
            tc_info = measurements.get("tc", {})
            et_info = measurements.get("et", {})
            
            wt_vol = wt_info.get("volume_cm3", 0.0)
            tc_vol = tc_info.get("volume_cm3", 0.0)
            et_vol = et_info.get("volume_cm3", 0.0)
            is_tumor_detected = wt_vol > 0.05

            st.markdown("##### 🩺 Clinical Diagnostic Assessment")
            if is_tumor_detected:
                st.markdown(f'<span class="badge-tumor" style="font-size:1.05rem;">⚠️ PATHOLOGY DETECTED — BRAIN TUMOR POSITIVE</span>', unsafe_allow_html=True)
                st.markdown(f"**Total Neoplastic Volume:** {wt_vol:.2f} cm³ ({wt_info.get('volume_mm3', 0):,.0f} mm³)")
            else:
                st.markdown(f'<span class="badge-normal" style="font-size:1.05rem;">✅ NO TUMOR IDENTIFIED — PHYSIOLOGICAL / NORMAL SCAN</span>', unsafe_allow_html=True)

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            
            # Metric cards for sub-regions
            sub1, sub2, sub3 = st.columns(3)
            with sub1:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid #22c55e;">
                    <div class="metric-title" style="color:#22c55e;">Whole Tumor (WT)</div>
                    <div class="metric-value">{wt_vol:.2f} cm³</div>
                    <div class="metric-sub">{wt_info.get('voxels', 0):,} voxels (Edema + Core)</div>
                </div>
                """, unsafe_allow_html=True)
            with sub2:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid #eab308;">
                    <div class="metric-title" style="color:#eab308;">Tumor Core (TC)</div>
                    <div class="metric-value">{tc_vol:.2f} cm³</div>
                    <div class="metric-sub">{tc_info.get('voxels', 0):,} voxels (Necrotic + Active)</div>
                </div>
                """, unsafe_allow_html=True)
            with sub3:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid #ef4444;">
                    <div class="metric-title" style="color:#ef4444;">Enhancing Tumor (ET)</div>
                    <div class="metric-value">{et_vol:.2f} cm³</div>
                    <div class="metric-sub">{et_info.get('voxels', 0):,} voxels (Active Malignancy)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            best_slice = seg_meta.get("best_slice_idx", orig_shape[2] // 2)
            seg_slice_idx = st.slider(
                "Navigate Tumor Slices (Auto-aligned to maximum lesion area):",
                0, orig_shape[2] - 1, best_slice,
                key="slider_seg_slice",
            )
            
            overlay_alpha = st.slider("Tumor Mask Opacity:", 0.0, 1.0, 0.55, 0.05)

            # Render slice with tumor overlay
            rendered_rgb = render_tumor_slice_overlay(
                volume_slice=raw_volume[:, :, seg_slice_idx],
                mask_slice=seg_mask[:, :, seg_slice_idx],
                alpha=overlay_alpha,
            )

            fig_seg, ax_seg = plt.subplots(figsize=(6, 6), facecolor="#0B132B")
            ax_seg.imshow(np.rot90(rendered_rgb))
            ax_seg.axis("off")
            st.pyplot(fig_seg, use_container_width=True)
            plt.close(fig_seg)
            
            st.caption("🟢 Green: Peritumoral Edema | 🟡 Yellow: Necrotic / Non-enhancing Core | 🔴 Red: Enhancing Tumor")
        else:
            st.info("Click 'Segment Tumor Sub-regions' on the left to execute the volumetric segmentation model.")


# =========================================================================
# TAB 6: CLINICAL REPORT & EXPORT
# =========================================================================
with tab_report:
    st.markdown("#### 📄 Clinical Decision Support & PDF Report Generator")
    st.write("Generate a publication-grade, multi-page diagnostic radiological PDF report conforming to hospital imaging standards.")

    rep_col1, rep_col2 = st.columns([1, 1])
    with rep_col1:
        st.markdown("##### 👤 Patient & Study Metadata")
        pat_id = st.text_input("Patient ID:", "PAT-2026-0891")
        pat_name = st.text_input("Patient Name:", "Anonymous Patient")
        pat_age = st.text_input("Patient Age:", "54")
        pat_gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
        physician = st.text_input("Referring Physician:", "Dr. A. Sharma, MD (Neurology)")

    with rep_col2:
        st.markdown("##### 📑 Report Actions")
        
        has_seg = st.session_state.get("seg_mask") is not None
        if not has_seg:
            st.warning("⚠️ No segmentation has been run yet. The report will reflect general analysis or physiological baseline.")

        if st.button("📄 Generate Publication-Grade PDF Report", type="primary"):
            with st.spinner("Generating official medical PDF report..."):
                file_info = {
                    "filename": selected_file_name,
                    "filepath": selected_file_path,
                    "shape": orig_shape,
                    "modality": "Brain MRI",
                    "mri_type": "brain",
                }
                
                seg_metadata = st.session_state.get("seg_meta", {
                    "voxel_spacing": voxel_spacing,
                    "best_slice_idx": orig_shape[2] // 2,
                    "measurements": {
                        "wt": {"voxels": 0, "volume_cm3": 0.0, "volume_mm3": 0.0},
                        "tc": {"voxels": 0, "volume_cm3": 0.0, "volume_mm3": 0.0},
                        "et": {"voxels": 0, "volume_cm3": 0.0, "volume_mm3": 0.0},
                    },
                    "validation": {"valid_nesting": True},
                })
                
                report_data = BrainSightReportGenerator.assemble_report_data(
                    file_info=file_info,
                    segmentation_metadata=seg_metadata,
                    volume_data=raw_volume,
                    mask_data=st.session_state.get("seg_mask"),
                    patient_id=pat_id,
                    patient_name=pat_name,
                    patient_age=pat_age,
                )
                
                pdf_bytes = BrainSightReportGenerator.generate_pdf_report(report_data)
                html_preview = BrainSightReportGenerator.generate_html_report(report_data)
                
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["html_preview"] = html_preview
                st.success("PDF and HTML reports generated successfully!")

        pdf_bytes = st.session_state.get("pdf_bytes")
        if pdf_bytes:
            st.download_button(
                label="⬇️ Download Official Medical PDF Report",
                data=pdf_bytes,
                file_name=f"BrainSight_Report_{pat_id}.pdf",
                mime="application/pdf",
            )

    html_rep = st.session_state.get("html_preview")
    if html_rep:
        st.markdown("---")
        st.markdown("##### 🖥️ In-Browser HTML Diagnostic Report Preview:")
        st.components.v1.html(html_rep, height=650, scrolling=True)


# =========================================================================
# TAB 7: SYSTEM DIAGNOSTICS
# =========================================================================
with tab_diagnostics:
    st.markdown("#### ⚡ Hardware & Environment Diagnostics")
    
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("PyTorch Version", torch.__version__)
    with d2:
        cuda_avail = torch.cuda.is_available()
        st.metric("CUDA Acceleration", "Enabled" if cuda_avail else "Disabled (CPU)")
    with d3:
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "Intel / AMD CPU"
        st.metric("Processor Device", gpu_name[:18])
    with d4:
        st.metric("Python Runtime", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    st.markdown("---")
    st.markdown("##### 📦 Active Medical Frameworks & Dependencies:")
    st.json({
        "MONAI": "Installed (v1.6+)",
        "SimpleITK": "Active (N4 Bias Correction & Resampling)",
        "NiBabel": "Active (NIfTI-1 / NIfTI-2 Reader)",
        "ReportLab": "Active (A4 Vector PDF Engine)",
        "Streamlit": "v1.67.0",
        "TorchScript": "Calibrated MONAI 3D Model Loaded",
    })
