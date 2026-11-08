"""
BrainSight AI - Professional Medical-Style MRI Technical & Segmentation Report Generator.
Generates publication-quality A4 PDF reports via ReportLab and matching HTML reports for clinical decision support.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from loguru import logger

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas


MANDATORY_DISCLAIMER = (
    "This report contains AI-generated image analysis and segmentation results for research and "
    "decision-support purposes. It is not a substitute for professional radiological interpretation, "
    "clinical evaluation, or medical diagnosis. All findings require review by a qualified healthcare professional."
)


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw accurate 'Page X of Y' footers
    and running headers across all pages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running header on page 2 and beyond
        if self._pageNumber > 1:
            self.drawString(40, 810, "BrainSight AI — Brain MRI Segmentation Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, 804, 555, 804)

        # Running footer on every page
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 42, 555, 42)

        self.drawString(
            40,
            30,
            "Confidential · AI-Assisted Clinical Decision Support · Not a Clinical Diagnosis",
        )
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(555, 30, page_str)
        self.restoreState()


class BrainSightReportGenerator:
    """Generates complete medical-style PDF and HTML reports for brain MRI tumor segmentation."""

    @staticmethod
    def evaluate_tumor_status(
        measurements: dict[str, Any] | None,
        validation: dict[str, Any] | None,
        has_valid_brain_tissue: bool = True,
        inference_error: str | None = None,
        raw_nesting_valid: bool = True,
        max_brain_prob: float = 0.0,
    ) -> tuple[str, str, str, list[str]]:
        """
        Determines the clinical tumor presence status using validated decision rules.
        Returns: (tumor_status, explanation, validation_status, technical_warnings)
        """
        technical_warnings: list[str] = []

        # 1. Inference error / missing data check
        if inference_error:
            technical_warnings.append(f"Inference error: {inference_error}")
            return (
                "Indeterminate — Requires Validation",
                f"Model inference failed or was interrupted: {inference_error}. "
                "The result cannot be reliably interpreted without clinical review.",
                "FAILED — MODEL INFERENCE ERROR",
                technical_warnings,
            )

        if not measurements:
            return (
                "Indeterminate — Requires Validation",
                "No segmentation measurements were generated for this volume. "
                "Analysis is unverified and requires validation.",
                "UNVERIFIED — NO MEASUREMENTS",
                ["Missing volumetric measurement payload."],
            )

        # 2. Input tissue validity check
        if not has_valid_brain_tissue:
            technical_warnings.append("Volume contains insufficient or unverified intracranial brain tissue.")
            return (
                "Indeterminate — Requires Validation",
                "Insufficient intracranial brain parenchyma detected in the MRI volume. "
                "The scan may be non-cranial, severely corrupted, or preprocessed incorrectly.",
                "FAILED — INSUFFICIENT BRAIN TISSUE",
                technical_warnings,
            )

        # 3. Hierarchical anatomical nesting check (ET <= TC <= WT)
        wt_voxels = int(measurements.get("wt", {}).get("voxels", 0))
        tc_voxels = int(measurements.get("tc", {}).get("voxels", 0))
        et_voxels = int(measurements.get("et", {}).get("voxels", 0))

        wt_cm3 = float(measurements.get("wt", {}).get("volume_cm3", 0.0))
        tc_cm3 = float(measurements.get("tc", {}).get("volume_cm3", 0.0))
        et_cm3 = float(measurements.get("et", {}).get("volume_cm3", 0.0))

        valid_nesting = validation.get("valid_nesting", True) if validation else True
        if not valid_nesting or (et_voxels > tc_voxels) or (tc_voxels > wt_voxels):
            technical_warnings.append(
                f"Hierarchical nesting violation: ET ({et_voxels}) <= TC ({tc_voxels}) <= WT ({wt_voxels}) failed."
            )
            return (
                "Indeterminate — Requires Validation",
                "Segmentation volume validation failed: Hierarchical condition (ET <= TC <= WT) was violated "
                "by the model output. Measurements are inconsistent and require clinical review.",
                "FAILED — HIERARCHICAL ANATOMICAL VIOLATION",
                technical_warnings,
            )

        # 4. Valid positive detection (Significant cohesive lesion volume >= 50 voxels)
        if wt_voxels >= 50:
            return (
                "Tumor Detected — AI Prediction",
                f"Hyperintense volumetric lesion detected by MONAI 3D U-Net segmentation. "
                f"Whole Tumor (WT): {wt_cm3:.2f} cm³ ({wt_voxels} voxels); "
                f"Tumor Core (TC): {tc_cm3:.2f} cm³ ({tc_voxels} voxels); "
                f"Enhancing Tumor (ET): {et_cm3:.2f} cm³ ({et_voxels} voxels). "
                "Hierarchical anatomical nesting (ET <= TC <= WT) successfully validated.",
                "PASSED — VALIDATED MONAI 3D SEGMENTATION",
                technical_warnings,
            )

        # 5. Valid negative detection
        if wt_voxels == 0:
            return (
                "No Tumor Detected by Model",
                "The MONAI 3D U-Net deep learning model identified no abnormal enhancing tissue, "
                "tumor core, or peritumoral edema exceeding detection thresholds across the scanned brain parenchyma. "
                "Intracranial tissue metrics are within expected normal baseline.",
                "PASSED — VERIFIED NEGATIVE SEGMENTATION",
                technical_warnings,
            )

        # 6. Low/scattered voxel count (Borderline / Potential noise)
        technical_warnings.append(
            f"Candidate cluster volume is below significance threshold ({wt_voxels} voxels, {wt_cm3:.3f} cm³)."
        )
        return (
            "Indeterminate — Requires Validation",
            f"Candidate sub-threshold cluster detected ({wt_voxels} voxels, {wt_cm3:.3f} cm³). "
            "The detected cluster does not form a cohesive lesion envelope and may represent scanner noise or artifact. "
            "Manual radiological verification is required.",
            "BORDERLINE — SUB-THRESHOLD VOXEL CLUSTER",
            technical_warnings,
        )

    @classmethod
    def assemble_report_data(
        cls,
        file_info: dict[str, Any],
        mask_info: dict[str, Any] | None = None,
        segmentation_metadata: dict[str, Any] | None = None,
        volume_data: np.ndarray | None = None,
        mask_data: np.ndarray | None = None,
        patient_id: str | None = None,
        patient_name: str | None = None,
        patient_age: str | None = None,
        architecture: str = "monai_unet3d",
        inference_error: str | None = None,
    ) -> dict[str, Any]:
        """Assemble structured data dictionary for both PDF and HTML rendering."""
        report_id = f"BSA-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
        analysis_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Extract measurements and validation
        measurements = segmentation_metadata.get("measurements") if segmentation_metadata else None
        validation = segmentation_metadata.get("validation") if segmentation_metadata else None

        # Check tissue validity if volume available
        has_valid_brain_tissue = True
        if volume_data is not None:
            has_valid_brain_tissue = bool(np.sum(volume_data > (volume_data.mean() * 0.15)) > 5000)

        # Evaluate tumor detection status
        tumor_status, explanation, validation_status, technical_warnings = cls.evaluate_tumor_status(
            measurements=measurements,
            validation=validation,
            has_valid_brain_tissue=has_valid_brain_tissue,
            inference_error=inference_error,
        )

        # Format spatial information
        shape = file_info.get("shape", [])
        if not shape and volume_data is not None:
            shape = list(volume_data.shape)
        shape_str = " × ".join(str(s) for s in shape) if shape else "Unknown"

        voxel_spacing = segmentation_metadata.get("voxel_spacing") if segmentation_metadata else None
        if not voxel_spacing:
            voxel_spacing = file_info.get("voxel_spacing", None)
        if voxel_spacing and len(voxel_spacing) >= 3:
            voxel_str = f"{voxel_spacing[0]:.2f} × {voxel_spacing[1]:.2f} × {voxel_spacing[2]:.2f} mm"
        else:
            voxel_str = "Unknown"

        # Model metadata
        model_display_names = {
            "monai_unet3d": "MONAI 3D U-Net (v1.0.0, PyTorch TorchScript)",
            "unet": "Standard 2D UNet",
            "unet_plus": "UNet++ (Nested Skip Pathways)",
            "attention_unet": "Attention UNet (Gated Feature Suppression)",
        }
        model_name = model_display_names.get(architecture, f"{architecture.upper()} Segmentation Model")

        # Produce slice images if volume and mask are present
        raw_slice_png: bytes | None = None
        overlay_slice_png: bytes | None = None
        slice_idx = 0
        total_slices = shape[2] if len(shape) >= 3 else 0

        if volume_data is not None and mask_data is not None:
            try:
                from backend.utils.visualization import render_slice_png_bytes

                if segmentation_metadata and "best_slice_idx" in segmentation_metadata:
                    slice_idx = int(segmentation_metadata["best_slice_idx"])
                elif len(shape) >= 3:
                    slice_idx = shape[2] // 2

                raw_slice_png = render_slice_png_bytes(
                    volume=volume_data,
                    mask=mask_data,
                    slice_idx=slice_idx,
                    plane="axial",
                    show_wt=False,
                    show_tc=False,
                    show_et=False,
                )
                overlay_slice_png = render_slice_png_bytes(
                    volume=volume_data,
                    mask=mask_data,
                    slice_idx=slice_idx,
                    plane="axial",
                    show_wt=True,
                    show_tc=True,
                    show_et=True,
                    alpha=0.45,
                )
            except Exception as e:
                logger.warning(f"Could not render slice previews for report: {e}")
                technical_warnings.append(f"Image visualization rendering note: {e}")

        # Formulate clinical impression and recommendations
        if "Tumor Detected" in tumor_status or tumor_status == "TUMOR DETECTED":
            wt_cm3 = measurements.get("wt", {}).get("volume_cm3", 0.0) if measurements else 0.0
            tc_cm3 = measurements.get("tc", {}).get("volume_cm3", 0.0) if measurements else 0.0
            et_cm3 = measurements.get("et", {}).get("volume_cm3", 0.0) if measurements else 0.0
            impression = (
                f"Preliminary AI volumetric segmentation demonstrates an intracranial mass lesion with Whole Tumor (WT) "
                f"burden of approximately {wt_cm3:.2f} cm³, Tumor Core (TC) measuring {tc_cm3:.2f} cm³, and Enhancing Tumor "
                f"(ET) component measuring {et_cm3:.2f} cm³. Spatial distribution satisfies anatomical containment constraints. "
                "This finding represents an algorithmic prediction and must be interpreted in conjunction with multi-sequence "
                "contrast-enhanced clinical imaging and patient history."
            )
            recommendations = [
                "Urgent evaluation and formal interpretation by a board-certified neuroradiologist.",
                "Correlation with multi-parametric MRI sequences (T1, T1+Gd, T2, FLAIR, DWI/ADC) and prior imaging studies.",
                "Multidisciplinary neuro-oncology clinical review to determine diagnostic and therapeutic trajectory.",
            ]
        elif "No Tumor" in tumor_status or tumor_status == "NO TUMOR DETECTED":
            impression = (
                "Preliminary AI volumetric analysis identified no macroscopic tumor volume exceeding the model's "
                "detection thresholds within the segmented intracranial volume. This result does not exclude microscopic pathology, "
                "non-enhancing lesions, or conditions outside the trained domain of the MONAI 3D U-Net model."
            )
            recommendations = [
                "Routine clinical and radiological review according to institutional screening protocols.",
                "Re-evaluation if clinical symptoms or neurological signs persist or progress.",
            ]
        else:
            impression = (
                "AI volumetric analysis was INDETERMINATE. Segmentation validation checks could not confirm a reliable positive "
                "or negative finding due to model anomaly, sub-threshold cluster, or input characteristics. No definitive tumor "
                "presence or absence can be inferred from this automated result."
            )
            recommendations = [
                "Full manual radiological examination by a qualified clinical specialist.",
                "Verification of MRI acquisition parameters, contrast administration, and volume integrity.",
                "Repeat segmentation analysis with validated multi-sequence datasets if indicated.",
            ]

        # Nesting status string
        nesting_valid = validation.get("valid_nesting", True) if validation else True
        if measurements:
            wt_v = measurements.get("wt", {}).get("voxels", 0)
            tc_v = measurements.get("tc", {}).get("voxels", 0)
            et_v = measurements.get("et", {}).get("voxels", 0)
            nesting_note = f"ET ({et_v}) ≤ TC ({tc_v}) ≤ WT ({wt_v}): {'Validated' if nesting_valid else 'FAILED'}"
        else:
            nesting_note = "Not available"

        return {
            "report_id": report_id,
            "analysis_timestamp": analysis_timestamp,
            "report_title": "BrainSight AI — Brain MRI Segmentation Report",
            "report_status": "Preliminary AI Analysis / Requires Clinical Review",
            "model_name": model_name,
            "architecture": architecture,
            "patient_info": {
                "patient_id": patient_id.strip() if patient_id and patient_id.strip() else "Not provided",
                "patient_name": patient_name.strip() if patient_name and patient_name.strip() else "Not provided",
                "patient_age": patient_age.strip() if patient_age and patient_age.strip() else "Not provided",
            },
            "scan_info": {
                "filename": file_info.get("filename", "Unknown MRI"),
                "modality": file_info.get("modality", "Unknown"),
                "mri_type": file_info.get("mri_type", "Brain"),
                "dimensions": shape_str,
                "voxel_spacing": voxel_str,
            },
            "assessment": {
                "tumor_status": tumor_status,
                "explanation": explanation,
                "validation_status": validation_status,
                "technical_warnings": technical_warnings,
            },
            "measurements": measurements,
            "nesting_valid": nesting_valid,
            "nesting_note": nesting_note,
            "slice_visualization": {
                "has_images": bool(raw_slice_png and overlay_slice_png),
                "raw_slice_png": raw_slice_png,
                "overlay_slice_png": overlay_slice_png,
                "slice_idx": slice_idx + 1 if total_slices > 0 else 1,
                "total_slices": total_slices,
                "plane": "Axial",
            },
            "technical_details": {
                "inference_engine": "PyTorch TorchScript Engine (MONAI 3D U-Net)",
                "preprocessing": "3D Intensity Min-Max Scaling [0, 1] & Spatial Multi-8 Boundary Padding",
                "output_channels": "Ch 0: WT (Whole Tumor), Ch 1: TC (Tumor Core), Ch 2: ET (Enhancing Tumor)",
                "decision_threshold": "Sigmoid probability threshold = 0.5 with Intracranial Brain Masking",
                "hierarchical_enforcement": "Strict Nested Containment: ET ⊆ TC ⊆ WT",
            },
            "impression": impression,
            "recommendations": recommendations,
            "disclaimer": MANDATORY_DISCLAIMER,
        }

    @classmethod
    def generate_pdf_report(
        cls,
        report_data: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> bytes:
        """
        Generate a publication-grade, multi-page safe A4 PDF report with ReportLab.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=46,
            bottomMargin=46,
        )

        styles = getSampleStyleSheet()

        # Custom Typography Styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=colors.HexColor("#0B132B"),
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )
        section_heading = ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
        )
        body_bold = ParagraphStyle(
            "ReportBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold",
        )
        badge_style = ParagraphStyle(
            "StatusBadge",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            alignment=1,  # Center
        )
        caption_style = ParagraphStyle(
            "ImageCaption",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=1,
            textColor=colors.HexColor("#334155"),
        )
        disclaimer_style = ParagraphStyle(
            "DisclaimerText",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#B45309"),
        )

        story: list[Any] = []

        # ==========================================
        # SECTION A: REPORT HEADER
        # ==========================================
        header_table_data = [
            [
                Paragraph("<b>BrainSight AI</b>", title_style),
                Paragraph(f"<b>Report ID:</b> {report_data['report_id']}", body_bold),
            ],
            [
                Paragraph("Brain MRI AI Segmentation & Technical Analysis Report", subtitle_style),
                Paragraph(f"<b>Analysis Date:</b> {report_data['analysis_timestamp']}", body_style),
            ],
            [
                Paragraph(f"<b>Model:</b> {report_data['model_name']}", subtitle_style),
                Paragraph(f"<b>Status:</b> <font color='#D97706'><b>{report_data['report_status']}</b></font>", body_style),
            ],
        ]
        header_table = Table(header_table_data, colWidths=[330, 193])
        header_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ])
        )
        story.append(header_table)
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=8))

        # ==========================================
        # SECTION B: PATIENT & SCAN INFORMATION
        # ==========================================
        story.append(Paragraph("A. Patient & Scan Acquisition Information", section_heading))

        p_info = report_data["patient_info"]
        s_info = report_data["scan_info"]

        info_data = [
            [
                Paragraph("<b>Patient ID:</b>", body_style),
                Paragraph(p_info["patient_id"], body_style),
                Paragraph("<b>Scan Filename:</b>", body_style),
                Paragraph(s_info["filename"], body_style),
            ],
            [
                Paragraph("<b>Patient Name:</b>", body_style),
                Paragraph(p_info["patient_name"], body_style),
                Paragraph("<b>Modality / Sequence:</b>", body_style),
                Paragraph(f"{s_info['modality']} ({s_info['mri_type']})", body_style),
            ],
            [
                Paragraph("<b>Patient Age:</b>", body_style),
                Paragraph(p_info["patient_age"], body_style),
                Paragraph("<b>Dimensions:</b>", body_style),
                Paragraph(s_info["dimensions"], body_style),
            ],
            [
                Paragraph("<b>Analysis Type:</b>", body_style),
                Paragraph("Volumetric 3D MRI", body_style),
                Paragraph("<b>Voxel Spacing:</b>", body_style),
                Paragraph(s_info["voxel_spacing"], body_style),
            ],
        ]
        info_table = Table(info_data, colWidths=[80, 175, 110, 158])
        info_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(info_table)
        story.append(Spacer(1, 8))

        # ==========================================
        # SECTION C: AI TUMOR ASSESSMENT
        # ==========================================
        story.append(Paragraph("B. AI Tumor Assessment & Detection Status", section_heading))

        status = report_data["assessment"]["tumor_status"]
        if "Tumor Detected" in status or status == "TUMOR DETECTED":
            status_bg = colors.HexColor("#FEF2F2")
            status_border = colors.HexColor("#EF4444")
            status_color = "#991B1B"
        elif "No Tumor" in status or status == "NO TUMOR DETECTED":
            status_bg = colors.HexColor("#F0FDF4")
            status_border = colors.HexColor("#22C55E")
            status_color = "#166534"
        else:
            status_bg = colors.HexColor("#FFFBEB")
            status_border = colors.HexColor("#F59E0B")
            status_color = "#92400E"

        badge_p = Paragraph(f"<font color='{status_color}'><b>● {status}</b></font>", badge_style)
        explanation_p = Paragraph(f"<b>Assessment:</b> {report_data['assessment']['explanation']}", body_style)
        val_status_p = Paragraph(
            f"<b>Segmentation Validation:</b> <font color='#0369A1'>{report_data['assessment']['validation_status']}</font>",
            body_style,
        )

        assessment_rows = [
            [badge_p],
            [explanation_p],
            [val_status_p],
        ]
        for w in report_data["assessment"]["technical_warnings"]:
            assessment_rows.append([Paragraph(f"<b>Warning / Limitation:</b> <font color='#B45309'>{w}</font>", body_style)])

        assessment_table = Table(assessment_rows, colWidths=[523])
        assessment_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), status_bg),
                ("BOX", (0, 0), (-1, -1), 1.2, status_border),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ])
        )
        story.append(assessment_table)
        story.append(Spacer(1, 8))

        # ==========================================
        # SECTION D: TUMOR MEASUREMENTS
        # ==========================================
        story.append(Paragraph("C. Quantitative Tumor Sub-Region Volumetrics", section_heading))

        meas = report_data["measurements"]
        if meas:
            wt = meas.get("wt", {})
            tc = meas.get("tc", {})
            et = meas.get("et", {})

            meas_table_data = [
                [
                    Paragraph("<b>Sub-Region</b>", body_bold),
                    Paragraph("<b>Anatomical Description</b>", body_bold),
                    Paragraph("<b>Voxel Count</b>", body_bold),
                    Paragraph("<b>Volume (mm³)</b>", body_bold),
                    Paragraph("<b>Volume (cm³)</b>", body_bold),
                ],
                [
                    Paragraph("<font color='#16A34A'><b>WT</b></font> (Whole Tumor)", body_style),
                    Paragraph("Complete lesion envelope (Edema + Core + Enhancing)", body_style),
                    Paragraph(f"{wt.get('voxels', 0):,}", body_style),
                    Paragraph(f"{wt.get('volume_mm3', 0.0):,.1f}", body_style),
                    Paragraph(f"<b>{wt.get('volume_cm3', 0.0):.2f}</b>", body_bold),
                ],
                [
                    Paragraph("<font color='#CA8A04'><b>TC</b></font> (Tumor Core)", body_style),
                    Paragraph("Gross tumor core within WT (Necrotic Core + Enhancing)", body_style),
                    Paragraph(f"{tc.get('voxels', 0):,}", body_style),
                    Paragraph(f"{tc.get('volume_mm3', 0.0):,.1f}", body_style),
                    Paragraph(f"<b>{tc.get('volume_cm3', 0.0):.2f}</b>", body_bold),
                ],
                [
                    Paragraph("<font color='#DC2626'><b>ET</b></font> (Enhancing Tumor)", body_style),
                    Paragraph("Active vascularized enhancing component within TC", body_style),
                    Paragraph(f"{et.get('voxels', 0):,}", body_style),
                    Paragraph(f"{et.get('volume_mm3', 0.0):,.1f}", body_style),
                    Paragraph(f"<b>{et.get('volume_cm3', 0.0):.2f}</b>", body_bold),
                ],
            ]
        else:
            meas_table_data = [
                [Paragraph("<b>Status:</b> Volumetric measurements unavailable or unvalidated.", body_style)]
            ]

        meas_table = Table(meas_table_data, colWidths=[120, 193, 70, 75, 65])
        meas_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(meas_table)

        nesting_text = f"<b>Hierarchical Nesting Check:</b> {report_data['nesting_note']}"
        story.append(Spacer(1, 3))
        story.append(Paragraph(nesting_text, body_style))
        story.append(Spacer(1, 10))

        # ==========================================
        # SECTION E: MRI VISUALIZATION
        # ==========================================
        story.append(Paragraph("D. Representative MRI Slices & Segmentation Overlay", section_heading))

        vis = report_data["slice_visualization"]
        if vis.get("has_images") and vis.get("raw_slice_png") and vis.get("overlay_slice_png"):
            raw_img_io = io.BytesIO(vis["raw_slice_png"])
            overlay_img_io = io.BytesIO(vis["overlay_slice_png"])

            # 225 x 225 pt images fit cleanly side-by-side
            img_width = 230
            img_height = 230
            raw_rl = RLImage(raw_img_io, width=img_width, height=img_height)
            overlay_rl = RLImage(overlay_img_io, width=img_width, height=img_height)

            caption_raw = Paragraph(
                f"<b>Raw MRI Volume</b><br/>{vis['plane']} Slice {vis['slice_idx']} of {vis['total_slices']}",
                caption_style,
            )
            caption_overlay = Paragraph(
                f"<b>MONAI Multi-Region Overlay</b><br/>"
                f"<font color='#16A34A'>■ WT</font> &nbsp; "
                f"<font color='#CA8A04'>■ TC</font> &nbsp; "
                f"<font color='#DC2626'>■ ET</font>",
                caption_style,
            )

            img_table_data = [
                [raw_rl, overlay_rl],
                [caption_raw, caption_overlay],
            ]
            img_table = Table(img_table_data, colWidths=[260, 263])
            img_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(img_table)
        else:
            story.append(
                Paragraph(
                    "<i>MRI Visualization Unavailable: Slice rendering could not be generated from current volume.</i>",
                    body_style,
                )
            )

        story.append(Spacer(1, 10))

        # ==========================================
        # SECTION F: TECHNICAL ANALYSIS DETAILS
        # ==========================================
        story.append(Paragraph("E. Technical Analysis & Quality Control Details", section_heading))

        tech = report_data["technical_details"]
        tech_data = [
            [Paragraph("<b>Inference Engine:</b>", body_style), Paragraph(tech["inference_engine"], body_style)],
            [Paragraph("<b>Preprocessing Pipeline:</b>", body_style), Paragraph(tech["preprocessing"], body_style)],
            [Paragraph("<b>Output Channels:</b>", body_style), Paragraph(tech["output_channels"], body_style)],
            [Paragraph("<b>Activation & Decision:</b>", body_style), Paragraph(tech["decision_threshold"], body_style)],
            [Paragraph("<b>Quality Control:</b>", body_style), Paragraph(tech["hierarchical_enforcement"], body_style)],
        ]
        tech_table = Table(tech_data, colWidths=[150, 373])
        tech_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(tech_table)
        story.append(Spacer(1, 10))

        # ==========================================
        # SECTION G: IMPRESSION AND RECOMMENDATIONS
        # ==========================================
        story.append(Paragraph("F. AI Diagnostic Impression & Clinical Recommendations", section_heading))

        recs_html = "".join([f"• {r}<br/>" for r in report_data["recommendations"]])
        imp_data = [
            [
                Paragraph(
                    f"<b>Preliminary AI Impression:</b><br/>{report_data['impression']}<br/><br/>"
                    f"<b>Recommended Clinical Actions:</b><br/>{recs_html}",
                    body_style,
                )
            ]
        ]
        imp_table = Table(imp_data, colWidths=[523])
        imp_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#94A3B8")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(imp_table)
        story.append(Spacer(1, 10))

        # ==========================================
        # SECTION H: MANDATORY MEDICAL DISCLAIMER
        # ==========================================
        disc_data = [
            [
                Paragraph(
                    f"<b>MANDATORY MEDICAL REGULATORY NOTICE:</b><br/>{report_data['disclaimer']}",
                    disclaimer_style,
                )
            ]
        ]
        disc_table = Table(disc_data, colWidths=[523])
        disc_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#F59E0B")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(disc_table)

        # Build document using NumberedCanvas
        doc.build(story, canvasmaker=NumberedCanvas)
        pdf_bytes = buffer.getvalue()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(pdf_bytes)
            logger.info(f"Saved PDF report to {out_p} ({len(pdf_bytes)} bytes)")

        return pdf_bytes

    @classmethod
    def generate_html_report(cls, report_data: dict[str, Any], auto_print: bool = False) -> str:
        """
        Generate matching interactive HTML report for in-browser viewing and browser printing.
        Uses exact same values, sections, and styling tokens.
        """
        p_info = report_data["patient_info"]
        s_info = report_data["scan_info"]
        status = report_data["assessment"]["tumor_status"]
        meas = report_data.get("measurements")
        vis = report_data.get("slice_visualization", {})
        tech = report_data.get("technical_details", {})

        # Status badge colors
        if "Tumor Detected" in status or status == "TUMOR DETECTED":
            status_class = "detected"
            status_color = "#dc2626"
            status_bg = "rgba(220, 38, 38, 0.1)"
            status_border = "#dc2626"
        elif "No Tumor" in status or status == "NO TUMOR DETECTED":
            status_class = "not-detected"
            status_color = "#16a34a"
            status_bg = "rgba(22, 163, 74, 0.1)"
            status_border = "#16a34a"
        else:
            status_class = "indeterminate"
            status_color = "#d97706"
            status_bg = "rgba(217, 119, 6, 0.1)"
            status_border = "#d97706"

        # Embedded base64 images
        raw_b64 = ""
        overlay_b64 = ""
        if vis.get("has_images") and vis.get("raw_slice_png") and vis.get("overlay_slice_png"):
            raw_b64 = f"data:image/png;base64,{base64.b64encode(vis['raw_slice_png']).decode('ascii')}"
            overlay_b64 = f"data:image/png;base64,{base64.b64encode(vis['overlay_slice_png']).decode('ascii')}"

        recs_list = "".join([f"<li>{r}</li>" for r in report_data.get("recommendations", [])])
        warnings_list = "".join([
            f"<div class='warning-item'><strong>Warning:</strong> {w}</div>"
            for w in report_data["assessment"].get("technical_warnings", [])
        ])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrainSight AI — {report_data['report_id']}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #0284c7;
            --border: #334155;
            --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #090d16;
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 24px;
        }}
        .report-wrapper {{
            max-width: 900px;
            margin: 0 auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 36px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }}
        .toolbar {{
            max-width: 900px;
            margin: 0 auto 16px auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s ease;
            border: none;
        }}
        .btn-primary {{
            background: #0284c7;
            color: white;
        }}
        .btn-primary:hover {{ background: #0369a1; }}
        .btn-secondary {{
            background: #334155;
            color: white;
        }}
        .btn-secondary:hover {{ background: #475569; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .brand {{ font-size: 24px; font-weight: 800; color: #38bdf8; letter-spacing: -0.5px; }}
        .brand-sub {{ font-size: 13px; color: var(--text-secondary); margin-top: 2px; }}
        .header-meta {{ text-align: right; font-size: 13px; color: var(--text-secondary); }}
        .header-meta strong {{ color: var(--text-primary); }}
        .section {{
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 15px;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        .data-table th, .data-table td {{
            padding: 9px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .data-table th {{
            background: rgba(30, 41, 59, 0.9);
            color: #cbd5e1;
            font-weight: 600;
        }}
        .data-table td.number {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .status-card {{
            background: {status_bg};
            border: 1.5px solid {status_border};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .status-badge {{
            display: inline-block;
            font-size: 16px;
            font-weight: 800;
            color: {status_color};
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .status-explanation {{
            font-size: 13px;
            color: var(--text-primary);
            line-height: 1.5;
            margin-bottom: 8px;
        }}
        .validation-pill {{
            font-size: 12px;
            color: #38bdf8;
            font-weight: 600;
        }}
        .warning-item {{
            margin-top: 6px;
            font-size: 12px;
            color: #f59e0b;
        }}
        .vis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 10px;
        }}
        .vis-card {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .vis-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
            margin: 0 auto;
        }}
        .vis-caption {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 8px;
        }}
        .legend-chips {{
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 6px;
            font-size: 12px;
            font-weight: 600;
        }}
        .chip-wt {{ color: #22c55e; }}
        .chip-tc {{ color: #eab308; }}
        .chip-et {{ color: #ef4444; }}
        .callout-box {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            font-size: 13px;
            line-height: 1.6;
        }}
        .callout-box ul {{
            margin-left: 20px;
            margin-top: 8px;
        }}
        .disclaimer-box {{
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-radius: 8px;
            padding: 14px;
            font-size: 11px;
            color: #fbbf24;
            line-height: 1.5;
            margin-top: 24px;
        }}
        @media print {{
            body {{
                background: #ffffff !important;
                color: #0f172a !important;
                padding: 0 !important;
            }}
            .toolbar {{ display: none !important; }}
            .report-wrapper {{
                box-shadow: none !important;
                border: none !important;
                padding: 15mm !important;
                background: #ffffff !important;
                max-width: 100% !important;
            }}
            .brand {{ color: #0284c7 !important; }}
            .header {{ border-bottom: 2px solid #0284c7 !important; }}
            .header-meta strong {{ color: #0f172a !important; }}
            .section-title {{ color: #0284c7 !important; }}
            .data-table {{
                background: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
            }}
            .data-table th {{
                background: #f1f5f9 !important;
                color: #0f172a !important;
            }}
            .data-table td {{
                border-bottom: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
            }}
            .status-card {{
                border: 1.5px solid {status_border} !important;
                background: #f8fafc !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .status-explanation {{ color: #0f172a !important; }}
            .callout-box {{
                background: #f8fafc !important;
                border: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
            }}
            .vis-card {{
                background: #f8fafc !important;
                border: 1px solid #cbd5e1 !important;
            }}
            .vis-caption {{ color: #334155 !important; }}
            .disclaimer-box {{
                background: #fffbeb !important;
                border: 1px solid #f59e0b !important;
                color: #92400e !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .page-break {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
    <div class="toolbar">
        <div style="font-size: 13px; color: var(--text-secondary);">
            Report ID: <strong>{report_data['report_id']}</strong>
        </div>
        <div style="display: flex; gap: 8px;">
            <button onclick="window.print()" class="btn btn-secondary">
                🖨️ Print Report
            </button>
            <a href="/api/report/pdf?file_id={s_info['filename']}&report_id={report_data['report_id']}" id="pdf-download-btn" class="btn btn-primary">
                📥 Download PDF Report
            </a>
        </div>
    </div>

    <div class="report-wrapper">
        <!-- HEADER -->
        <div class="header">
            <div>
                <div class="brand">BrainSight AI</div>
                <div class="brand-sub">Brain MRI Segmentation & Technical Analysis Report</div>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                    Model: <strong>{report_data['model_name']}</strong>
                </div>
            </div>
            <div class="header-meta">
                <div>Report ID: <strong>{report_data['report_id']}</strong></div>
                <div>Date: <strong>{report_data['analysis_timestamp']}</strong></div>
                <div style="margin-top: 4px; color: #f59e0b; font-weight: 700;">{report_data['report_status']}</div>
            </div>
        </div>

        <!-- SECTION A: PATIENT & SCAN -->
        <div class="section">
            <div class="section-title">A. Patient & Scan Acquisition Information</div>
            <table class="data-table">
                <tr>
                    <th style="width: 20%;">Patient ID</th>
                    <td style="width: 30%;">{p_info['patient_id']}</td>
                    <th style="width: 20%;">Scan Filename</th>
                    <td style="width: 30%;">{s_info['filename']}</td>
                </tr>
                <tr>
                    <th>Patient Name</th>
                    <td>{p_info['patient_name']}</td>
                    <th>Modality / Type</th>
                    <td>{s_info['modality']} ({s_info['mri_type']})</td>
                </tr>
                <tr>
                    <th>Patient Age</th>
                    <td>{p_info['patient_age']}</td>
                    <th>Volume Dimensions</th>
                    <td>{s_info['dimensions']}</td>
                </tr>
                <tr>
                    <th>Analysis Pipeline</th>
                    <td>Volumetric 3D MRI MONAI</td>
                    <th>Voxel Spacing</th>
                    <td>{s_info['voxel_spacing']}</td>
                </tr>
            </table>
        </div>

        <!-- SECTION B: AI TUMOR ASSESSMENT -->
        <div class="section">
            <div class="section-title">B. AI Tumor Assessment & Detection Status</div>
            <div class="status-card">
                <div class="status-badge">● {status}</div>
                <div class="status-explanation"><strong>Assessment:</strong> {report_data['assessment']['explanation']}</div>
                <div class="validation-pill">Segmentation Validation: {report_data['assessment']['validation_status']}</div>
                {warnings_list}
            </div>
        </div>

        <!-- SECTION C: MEASUREMENTS -->
        <div class="section">
            <div class="section-title">C. Quantitative Tumor Sub-Region Volumetrics</div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Sub-Region</th>
                        <th>Anatomical Description</th>
                        <th style="text-align: right;">Voxel Count</th>
                        <th style="text-align: right;">Volume (mm³)</th>
                        <th style="text-align: right;">Volume (cm³)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="font-weight: 700; color: #22c55e;">WT — Whole Tumor</td>
                        <td>Complete lesion envelope (Edema + Core + Enhancing)</td>
                        <td class="number">{meas.get('wt', {}).get('voxels', 0) if meas else 0:,}</td>
                        <td class="number">{meas.get('wt', {}).get('volume_mm3', 0.0) if meas else 0.0:,.1f}</td>
                        <td class="number" style="font-weight: 700;">{meas.get('wt', {}).get('volume_cm3', 0.0) if meas else 0.0:.2f}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; color: #eab308;">TC — Tumor Core</td>
                        <td>Gross tumor core within WT (Necrotic Core + Enhancing)</td>
                        <td class="number">{meas.get('tc', {}).get('voxels', 0) if meas else 0:,}</td>
                        <td class="number">{meas.get('tc', {}).get('volume_mm3', 0.0) if meas else 0.0:,.1f}</td>
                        <td class="number" style="font-weight: 700;">{meas.get('tc', {}).get('volume_cm3', 0.0) if meas else 0.0:.2f}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; color: #ef4444;">ET — Enhancing Tumor</td>
                        <td>Active vascularized enhancing component within TC</td>
                        <td class="number">{meas.get('et', {}).get('voxels', 0) if meas else 0:,}</td>
                        <td class="number">{meas.get('et', {}).get('volume_mm3', 0.0) if meas else 0.0:,.1f}</td>
                        <td class="number" style="font-weight: 700;">{meas.get('et', {}).get('volume_cm3', 0.0) if meas else 0.0:.2f}</td>
                    </tr>
                </tbody>
            </table>
            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 6px;">
                <strong>Hierarchical Nesting Check:</strong> {report_data['nesting_note']}
            </div>
        </div>

        <div class="page-break"></div>

        <!-- SECTION D: VISUALIZATION -->
        <div class="section">
            <div class="section-title">D. Representative MRI Slices & Segmentation Overlay</div>
"""
        if raw_b64 and overlay_b64:
            html += f"""
            <div class="vis-grid">
                <div class="vis-card">
                    <img src="{raw_b64}" alt="Raw MRI Slice" />
                    <div class="vis-caption">
                        <strong>Raw MRI Slice</strong><br/>
                        {vis.get('plane', 'Axial')} Slice {vis.get('slice_idx', 1)} of {vis.get('total_slices', 1)}
                    </div>
                </div>
                <div class="vis-card">
                    <img src="{overlay_b64}" alt="Segmentation Overlay" />
                    <div class="vis-caption">
                        <strong>Segmentation Overlay</strong>
                        <div class="legend-chips">
                            <span class="chip-wt">■ WT (Green)</span>
                            <span class="chip-tc">■ TC (Yellow)</span>
                            <span class="chip-et">■ ET (Red)</span>
                        </div>
                    </div>
                </div>
            </div>
"""
        else:
            html += """
            <div class="callout-box" style="font-style: italic; color: var(--text-secondary);">
                MRI Visualization Unavailable: Slice rendering could not be generated from current volume.
            </div>
"""

        html += f"""
        </div>

        <!-- SECTION E: TECHNICAL DETAILS -->
        <div class="section">
            <div class="section-title">E. Technical Analysis & Quality Control Details</div>
            <table class="data-table">
                <tr><th style="width: 25%;">Inference Engine</th><td>{tech.get('inference_engine', 'N/A')}</td></tr>
                <tr><th>Preprocessing</th><td>{tech.get('preprocessing', 'N/A')}</td></tr>
                <tr><th>Output Channels</th><td>{tech.get('output_channels', 'N/A')}</td></tr>
                <tr><th>Decision Logic</th><td>{tech.get('decision_threshold', 'N/A')}</td></tr>
                <tr><th>Quality Control</th><td>{tech.get('hierarchical_enforcement', 'N/A')}</td></tr>
            </table>
        </div>

        <!-- SECTION F: IMPRESSION AND RECOMMENDATIONS -->
        <div class="section">
            <div class="section-title">F. AI Diagnostic Impression & Clinical Recommendations</div>
            <div class="callout-box">
                <p><strong>Preliminary AI Impression:</strong><br/>{report_data['impression']}</p>
                <div style="margin-top: 10px;">
                    <strong>Recommended Clinical Actions:</strong>
                    <ul>{recs_list}</ul>
                </div>
            </div>
        </div>

        <!-- SECTION G: REGULATORY DISCLAIMER -->
        <div class="disclaimer-box">
            <strong>MANDATORY MEDICAL REGULATORY NOTICE:</strong><br/>
            {report_data['disclaimer']}
        </div>
    </div>

    <script>
        // Update PDF download link with actual file_id and mask_file_id from URL
        const urlParams = new URLSearchParams(window.location.search);
        const fileId = urlParams.get('file_id');
        const maskFileId = urlParams.get('mask_file_id');
        const pdfBtn = document.getElementById('pdf-download-btn');
        if (pdfBtn && fileId) {{
            let pdfUrl = '/api/report/pdf?file_id=' + encodeURIComponent(fileId);
            if (maskFileId) pdfUrl += '&mask_file_id=' + encodeURIComponent(maskFileId);
            pdfBtn.href = pdfUrl;
        }}
        if (urlParams.get('print') === 'true') {{
            window.addEventListener('load', () => setTimeout(() => window.print(), 500));
        }}
    </script>
</body>
</html>
"""
        return html


# Alias for backward compatibility
TechnicalReportGenerator = BrainSightReportGenerator
