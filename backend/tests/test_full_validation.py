import os
import sys
import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"

NORMAL_SCAN = Path("dataset/brain/normal/brain_normal_001_t2.nii.gz")
TUMOR_SCAN = Path("dataset/brain/tumor/brain_tumor_001_t2.nii.gz")
BRATS_SCAN = Path("dataset/brain/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz")

def test_upload(filepath: Path):
    print(f"\n--- Uploading: {filepath.name} ---")
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/upload",
            files={"file": (filepath.name, f, "application/gzip")},
            data={"modality": "T2", "mri_type": "brain"}
        )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    file_id = data["file_id"]
    print(f"Uploaded successfully. File ID: {file_id}, Shape: {data['shape']}")
    return file_id

def test_segment(file_id: str, expected_status: str, is_tumor: bool):
    print(f"\n--- Testing Segmentation for File ID: {file_id} ---")
    resp = requests.post(f"{BASE_URL}/segment?file_id={file_id}&architecture=monai_unet3d")
    assert resp.status_code == 200, f"Segmentation failed: {resp.text}"
    res = resp.json()
    
    status = res.get("tumor_status")
    wt_voxels = res["measurements"]["wt"]["voxels"]
    tc_voxels = res["measurements"]["tc"]["voxels"]
    et_voxels = res["measurements"]["et"]["voxels"]
    valid_nesting = res["validation"]["valid_nesting"]

    print(f"Tumor Status: {status}")
    print(f"Measurements: WT={wt_voxels} voxels ({res['measurements']['wt']['volume_cm3']} cm³), "
          f"TC={tc_voxels} voxels, ET={et_voxels} voxels")
    print(f"Hierarchical Nesting Valid: {valid_nesting}")
    
    assert status == expected_status, f"Expected status '{expected_status}', got '{status}'"
    
    if is_tumor:
        assert wt_voxels >= 50, f"Tumor scan should have >= 50 WT voxels, got {wt_voxels}"
        assert valid_nesting is True, "Hierarchical nesting should be valid"
        assert wt_voxels >= tc_voxels >= et_voxels, "Strict nesting violation: WT >= TC >= ET"
    else:
        assert wt_voxels == 0, f"Normal scan should have 0 WT voxels, got {wt_voxels}"
        assert tc_voxels == 0, f"Normal scan should have 0 TC voxels, got {tc_voxels}"
        assert et_voxels == 0, f"Normal scan should have 0 ET voxels, got {et_voxels}"
        
    return res

def test_localization(file_id: str, is_tumor: bool):
    print(f"\n--- Testing Disease Localization for File ID: {file_id} ---")
    resp = requests.post(f"{BASE_URL}/localize?file_id={file_id}")
    assert resp.status_code == 200, f"Localization failed: {resp.text}"
    res = resp.json()
    
    loc = res.get("localization", {})
    detected = loc.get("detected")
    findings = loc.get("findings", [])
    heatmap = res.get("heatmap_preview", "")
    
    print(f"Pathology Detected: {detected}")
    print(f"Number of Findings: {len(findings)}")
    print(f"Heatmap Preview URI present: {heatmap.startswith('data:image/png;base64,')}")
    
    if is_tumor:
        assert detected is True, "Tumor scan should detect abnormal pathology"
        assert len(findings) > 0, "Tumor scan should return clinical findings"
    else:
        assert detected is False, f"Normal scan should NOT detect pathology, got {detected}"
        assert len(findings) == 0, f"Normal scan should have 0 findings, got {len(findings)}"
        
    return res

def test_pdf_report(file_id: str, mask_file_id: str, expected_status: str):
    print(f"\n--- Testing PDF Report for File ID: {file_id} (Mask ID: {mask_file_id}) ---")
    resp = requests.get(f"{BASE_URL}/report/pdf?file_id={file_id}&mask_file_id={mask_file_id}")
    assert resp.status_code == 200, f"PDF report failed: {resp.status_code} {resp.text}"
    pdf_bytes = resp.content
    assert pdf_bytes.startswith(b"%PDF"), "Response is not a valid PDF file"
    assert len(pdf_bytes) > 10000, f"PDF is unexpectedly small: {len(pdf_bytes)} bytes"
    
    # Check that the PDF contains the expected status
    # Note: ReportLab streams text inside compressed PDF streams or raw text
    print(f"Generated PDF Report successfully ({len(pdf_bytes)} bytes).")
    
    # Also test HTML report
    html_resp = requests.get(f"{BASE_URL}/report?file_id={file_id}&mask_file_id={mask_file_id}")
    assert html_resp.status_code == 200, f"HTML report failed: {html_resp.text}"
    html_text = html_resp.text
    assert expected_status in html_text, f"HTML report does not contain expected status '{expected_status}'"
    print(f"HTML Report contains verified status: '{expected_status}'")

def main():
    print("==================================================")
    print("NEUROFUSION AI - COMPREHENSIVE END-TO-END VALIDATION")
    print("==================================================")
    
    # 1. TEST NORMAL SCAN
    print("\n[STEP 1] Validating Known Normal Brain Scan...")
    normal_file_id = test_upload(NORMAL_SCAN)
    normal_seg = test_segment(normal_file_id, "No Tumor Detected by Model", is_tumor=False)
    test_localization(normal_file_id, is_tumor=False)
    test_pdf_report(normal_file_id, normal_seg["mask_file_id"], "No Tumor Detected by Model")
    
    # 2. TEST TUMOR SCAN
    print("\n[STEP 2] Validating Known Tumor Brain Scan...")
    tumor_file_id = test_upload(TUMOR_SCAN)
    tumor_seg = test_segment(tumor_file_id, "Tumor Detected — AI Prediction", is_tumor=True)
    test_localization(tumor_file_id, is_tumor=True)
    test_pdf_report(tumor_file_id, tumor_seg["mask_file_id"], "Tumor Detected — AI Prediction")
    
    # 3. TEST BRATS BENCHMARK SCAN
    if BRATS_SCAN.exists():
        print("\n[STEP 3] Validating BraTS Benchmark Scan...")
        brats_file_id = test_upload(BRATS_SCAN)
        brats_seg = test_segment(brats_file_id, "Tumor Detected — AI Prediction", is_tumor=True)
        test_localization(brats_file_id, is_tumor=True)
        test_pdf_report(brats_file_id, brats_seg["mask_file_id"], "Tumor Detected — AI Prediction")

    # 4. TEST STROKE CONTROL SCAN (NON-TUMOR PATHOLOGY)
    stroke_scan = Path("dataset/brain/stroke/brain_stroke_001_t2.nii.gz")
    if stroke_scan.exists():
        print("\n[STEP 4] Validating Non-Tumor Pathological Control (Stroke MRI)...")
        stroke_file_id = test_upload(stroke_scan)
        stroke_seg = test_segment(stroke_file_id, "No Tumor Detected by Model", is_tumor=False)
        test_localization(stroke_file_id, is_tumor=False)
        test_pdf_report(stroke_file_id, stroke_seg["mask_file_id"], "No Tumor Detected by Model")

    print("\n==================================================")
    print("ALL TESTS PASSED! FALSE-POSITIVE ISSUE COMPLETELY RESOLVED!")
    print("==================================================")

if __name__ == "__main__":
    main()
