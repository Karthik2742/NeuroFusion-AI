"""
NeuroFusion AI - API Schemas
Pydantic models for all request/response types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# === Enums ===

class MRIType(str, Enum):
    BRAIN = "brain"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    T1 = "T1"
    T1C = "T1c"
    T2 = "T2"
    FLAIR = "FLAIR"
    SEGMENTATION = "Segmentation"
    UNKNOWN = "Unknown"


class EnhancementModel(str, Enum):
    CLAHE = "clahe"
    RESTORMER = "restormer"
    SWINIR = "swinir"
    MIRNET = "mirnet"


class PreprocessingStep(str, Enum):
    BIAS_CORRECTION = "bias_correction"
    RESAMPLE = "resample"
    NORMALIZE = "normalize"
    SKULL_STRIP = "skull_strip"
    CROP = "crop"
    PAD = "pad"


class FilterType(str, Enum):
    CLAHE = "clahe"
    HISTOGRAM_EQ = "histogram_equalization"
    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    NLM = "non_local_means"
    ANISOTROPIC = "anisotropic_diffusion"


# === Request Schemas ===

class AnalyzeRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded file")
    analyze_full_dataset: bool = Field(False, description="Analyze entire dataset directory")
    dataset_path: Optional[str] = Field(None, description="Path to dataset directory")


class PreprocessRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded file")
    steps: list[PreprocessingStep] = Field(
        default=[PreprocessingStep.BIAS_CORRECTION, PreprocessingStep.NORMALIZE],
        description="Preprocessing steps to apply",
    )
    normalization_method: str = Field("zscore", description="Normalization method")
    save_output: bool = Field(True, description="Save preprocessed volume")


class EnhanceRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded file")
    model: EnhancementModel = Field(
        EnhancementModel.CLAHE,
        description="Enhancement model to use",
    )
    batch_size: int = Field(4, description="Batch size for inference")
    compute_metrics: bool = Field(True, description="Compute quality metrics")


class FilterRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded file")
    filter_type: FilterType = Field(..., description="Filter to apply")
    params: dict[str, Any] = Field(default_factory=dict, description="Filter parameters")


class EvaluateRequest(BaseModel):
    original_id: str = Field(..., description="ID of the original file")
    enhanced_id: str = Field(..., description="ID of the enhanced file")
    metrics: Optional[list[str]] = Field(None, description="Specific metrics to compute")


# === Response Schemas ===

class StatusResponse(BaseModel):
    status: str
    message: str
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_size_mb: float
    shape: list[int]
    modality: str
    mri_type: str
    preview: Optional[str] = None  # Base64 slice image


class AnalysisResponse(BaseModel):
    file_id: str
    mri_type: str
    modality: str
    intensity_stats: dict[str, Any]
    spatial_stats: dict[str, Any]
    quality_metrics: dict[str, float]
    charts: dict[str, str]  # Base64 chart images


class PreprocessResponse(BaseModel):
    file_id: str
    output_file_id: Optional[str] = None
    steps_applied: list[str]
    step_details: dict[str, Any]
    original_shape: list[int]
    output_shape: list[int]
    preview: Optional[str] = None


class EnhanceResponse(BaseModel):
    file_id: str
    model_used: str
    output_file_id: Optional[str] = None
    metrics: dict[str, float] = {}
    inference_time_sec: float = 0.0
    peak_memory_gb: float = 0.0
    device: str = "CPU"
    comparison_view: Optional[str] = None  # Base64 comparison image
    preview: Optional[str] = None


class EvaluateResponse(BaseModel):
    original_id: str
    enhanced_id: str
    enhancement_metrics: dict[str, float]
    charts: dict[str, str] = {}


class MetricsResponse(BaseModel):
    metrics: dict[str, float]
    charts: dict[str, str] = {}


class GPUStatusResponse(BaseModel):
    is_available: bool
    device_name: str
    total_memory_gb: float
    allocated_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    cuda_version: str


class DatasetStatsResponse(BaseModel):
    total_files: int
    analyzed_files: int
    type_distribution: dict[str, int]
    modality_distribution: dict[str, int]
    shape_stats: dict[str, Any]
    charts: dict[str, str] = {}
