"""
NeuroFusion AI - Test Runner (Standard Python unittest)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
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

SAMPLE_BRAIN_PATH = Path("./dataset/brain/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz")

class TestNeuroFusion(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_01_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_02_gpu_endpoint(self):
        response = self.client.get("/api/gpu")
        self.assertEqual(response.status_code, 200)
        self.assertIn("is_available", response.json())

    def test_03_models_endpoint(self):
        response = self.client.get("/api/models/enhancement")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertGreaterEqual(len(models), 4)

    def test_04_nifti_loader(self):
        self.assertTrue(SAMPLE_BRAIN_PATH.exists())
        data, header, affine = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
        self.assertEqual(data.ndim, 3)
        self.assertEqual(affine.shape, (4, 4))

    def test_05_dataset_analyzer(self):
        analyzer = DatasetAnalyzer()
        res = analyzer.analyze_file(SAMPLE_BRAIN_PATH)
        self.assertEqual(res["mri_type"], "brain")
        self.assertIn("intensity_stats", res)
        self.assertIn("quality_metrics", res)

    def test_06_preprocessor(self):
        preprocessor = MRIPreprocessor()
        res = preprocessor.run_pipeline(
            filepath=SAMPLE_BRAIN_PATH,
            steps=["resample", "normalize"]
        )
        self.assertIn("output_shape", res)
        self.assertEqual(len(res["steps_applied"]), 2)

    def test_07_enhancement_models(self):
        data, _, affine = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
        for model_name in ["clahe", "restormer", "swinir"]:
            model = get_enhancement_model(model_name)
            enhanced, meta = model.enhance_volume(data[:32, :32, :16])
            self.assertEqual(enhanced.shape, (32, 32, 16))
            self.assertGreaterEqual(meta["inference_time_sec"], 0)

    def test_08_enhancement_metrics(self):
        orig = np.random.rand(32, 32, 16).astype(np.float32)
        enh = orig + np.random.normal(0, 0.05, orig.shape).astype(np.float32)
        metrics = EnhancementMetrics.compute_all(orig, enh)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertGreater(metrics["psnr"], 0)

    def test_09_disease_localizer(self):
        data, _, _ = NIfTILoader.load_nibabel(SAMPLE_BRAIN_PATH)
        results = DiseaseLocalizer.localize_pathology(data, mri_type="brain")
        self.assertIn("findings", results)
        self.assertIn("abnormal_volume_ratio", results)

    def test_10_report_generator(self):
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
        
        report_data = TechnicalReportGenerator.assemble_report_data(
            file_info=file_info,
            volume_data=data,
        )
        html = TechnicalReportGenerator.generate_html_report(report_data)
        self.assertIn("Report", html)

    def test_11_full_api_workflow(self):
        # 1. Upload
        with open(SAMPLE_BRAIN_PATH, "rb") as f:
            resp = self.client.post("/api/upload", files={"file": ("test_flair.nii.gz", f, "application/gzip")})
        self.assertEqual(resp.status_code, 200)
        file_id = resp.json()["file_id"]

        # 2. Analyze
        resp = self.client.post("/api/analyze", json={"file_id": file_id})
        self.assertEqual(resp.status_code, 200)

        # 3. Preprocess
        resp = self.client.post("/api/preprocess", json={"file_id": file_id, "steps": ["normalize"]})
        self.assertEqual(resp.status_code, 200)

        # 4. Enhance
        resp = self.client.post("/api/enhance", json={"file_id": file_id, "model": "clahe", "compute_metrics": True})
        self.assertEqual(resp.status_code, 200)

        # 5. Localize
        resp = self.client.post(f"/api/localize?file_id={file_id}")
        self.assertEqual(resp.status_code, 200)

        # 6. Segment
        resp = self.client.post(f"/api/segment?file_id={file_id}&architecture=unet")
        self.assertEqual(resp.status_code, 200)

        # 7. Report
        resp = self.client.get(f"/api/report?file_id={file_id}")
        self.assertEqual(resp.status_code, 200)

    def test_12_dataset_tree_endpoint(self):
        resp = self.client.get("/api/dataset/tree")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("structure", data)
        self.assertIn("brain", data["structure"])
        self.assertIn("normal", data["structure"]["brain"])
        self.assertIn("tumor", data["structure"]["brain"])
        self.assertIn("stroke", data["structure"]["brain"])


if __name__ == "__main__":
    unittest.main()
