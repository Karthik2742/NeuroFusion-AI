"""
NeuroFusion AI - Comprehensive Test Suite
Validates backend components, ML pipelines, metrics, and FastAPI REST endpoints.
"""

try:
    import pytest
except ImportError:
    pytest = None
from pathlib import Path
import numpy as np
import nibabel as nib
from fastapi.testclient import TestClient

from backend.main import app
from backend.utils.io import NIfTILoader
from backend.preprocessing.dataset_analyzer import DatasetAnalyzer
from backend.preprocessing.preprocessor import MRIPreprocessor
from backend.models.enhancement import get_enhancement_model
from backend.models.segmentation import get_segmentation_model
from backend.evaluation.metrics import EnhancementMetrics
from backend.evaluation.explainability import DiseaseLocalizer
from backend.utils.report_generator import TechnicalReportGenerator

client = TestClient(app)
SAMPLE_BRAIN_PATH = Path("./dataset/brain/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_gpu_endpoint():
    response = client.get("/api/gpu")
    assert response.status_code == 200
    assert "is_available" in response.json()

def test_models_endpoint():
    response = client.get("/api/models/enhancement")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) >= 4

def test_nifti_loader():
    assert SAMPLE_BRAIN_PATH.exists(), "Sample NIfTI data missing"
    data, header, affine = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
    assert data.ndim == 3
    assert affine.shape == (4, 4)

def test_dataset_analyzer():
    analyzer = DatasetAnalyzer()
    res = analyzer.analyze_file(SAMPLE_BRAIN_PATH)
    assert res["mri_type"] == "brain"
    assert "intensity_stats" in res
    assert "quality_metrics" in res

def test_preprocessor():
    preprocessor = MRIPreprocessor()
    res = preprocessor.run_pipeline(
        filepath=SAMPLE_BRAIN_PATH,
        steps=["resample", "normalize"]
    )
    assert "output_shape" in res
    assert len(res["steps_applied"]) == 2

def test_enhancement_models():
    data, _, affine = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
    for model_name in ["clahe", "restormer", "swinir"]:
        model = get_enhancement_model(model_name)
        enhanced, meta = model.enhance_volume(data[:32, :32, :16])
        assert enhanced.shape == (32, 32, 16)
        assert meta["inference_time_sec"] >= 0

def test_enhancement_metrics():
    orig = np.random.rand(32, 32, 16).astype(np.float32)
    enh = orig + np.random.normal(0, 0.05, orig.shape).astype(np.float32)
    metrics = EnhancementMetrics.compute_all(orig, enh)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert metrics["psnr"] > 0

def test_disease_localizer():
    data, _, _ = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
    results = DiseaseLocalizer.localize_pathology(data, mri_type="brain")
    assert "findings" in results
    assert "abnormal_volume_ratio" in results

def test_report_generator():
    analyzer = DatasetAnalyzer()
    analysis_res = analyzer.analyze_file(SAMPLE_BRAIN_PATH)
    data, _, _ = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
    loc_res = DiseaseLocalizer.localize_pathology(data, mri_type="brain")
    
    file_info = {
        "filename": SAMPLE_BRAIN_PATH.name,
        "mri_type": "brain",
        "modality": "FLAIR",
        "shape": list(data.shape),
    }
    
    html = TechnicalReportGenerator.generate_html_report(file_info, analysis_res, loc_res)
    assert "<title>NeuroFusion AI - Technical Report</title>" in html

def test_upload_and_full_workflow():
    # 1. Upload
    with open(SAMPLE_BRAIN_PATH, "rb") as f:
        resp = client.post("/api/upload", files={"file": ("test_flair.nii.gz", f, "application/gzip")})
    assert resp.status_code == 200
    file_id = resp.json()["file_id"]
    assert file_id is not None

    # 2. Analyze
    resp = client.post("/api/analyze", json={"file_id": file_id})
    assert resp.status_code == 200
    assert resp.json()["file_id"] == file_id

    # 3. Preprocess
    resp = client.post("/api/preprocess", json={"file_id": file_id, "steps": ["normalize"]})
    assert resp.status_code == 200

    # 4. Enhance
    resp = client.post("/api/enhance", json={"file_id": file_id, "model": "clahe", "compute_metrics": True})
    assert resp.status_code == 200
    assert "metrics" in resp.json()

    # 5. Localize
    resp = client.post(f"/api/localize?file_id={file_id}")
    assert resp.status_code == 200

    # 6. Segment
    resp = client.post(f"/api/segment?file_id={file_id}&architecture=unet")
    assert resp.status_code == 200

    # 7. Report
    resp = client.get(f"/api/report?file_id={file_id}")
    assert resp.status_code == 200
    assert "NeuroFusion AI" in resp.text

if __name__ == "__main__":
    tests = [
        test_health_check,
        test_gpu_endpoint,
        test_models_endpoint,
        test_nifti_loader,
        test_dataset_analyzer,
        test_preprocessor,
        test_enhancement_models,
        test_enhancement_metrics,
        test_disease_localizer,
        test_report_generator,
        test_upload_and_full_workflow,
    ]
    for t in tests:
        print(f"Running {t.__name__}...", end=" ", flush=True)
        t()
        print("OK")
    print(f"All {len(tests)} tests passed successfully!")
