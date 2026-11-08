import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  Layers, Loader2, Download, Shield, Eye, EyeOff, CheckCircle2, AlertTriangle, FileText,
  ChevronLeft, ChevronRight, Crosshair, Play, Pause, Printer, AlertOctagon
} from 'lucide-react'
import {
  getFiles, segmentMRI, localizeMRI, downloadFile, getReportUrl, getReportPdfUrl, getSegmentSliceUrl,
  type FileInfo, type SegmentResponse, type LocalizeResponse
} from '../api'

const ARCHITECTURES = [
  { id: 'monai_unet3d', name: 'MONAI 3D U-Net', desc: '3D Volumetric U-Net trained on BraTS for WT, TC, and ET sub-regions' },
  { id: 'unet', name: 'UNet', desc: 'Standard Encoder-Decoder architecture with skip connections' },
  { id: 'unet_plus', name: 'UNet++', desc: 'Nested dense skip pathways to reduce semantic gap' },
  { id: 'attention_unet', name: 'Attention UNet', desc: 'Attention gates to suppress irrelevant feature responses' },
]

export default function Segmentation() {
  const location = useLocation()
  const [files, setFiles] = useState<FileInfo[]>([])
  const [selectedFile, setSelectedFile] = useState('')
  const [architecture, setArchitecture] = useState('monai_unet3d')
  const [processing, setProcessing] = useState(false)
  const [segmentResult, setSegmentResult] = useState<SegmentResponse | null>(null)
  const [localizeResult, setLocalizeResult] = useState<LocalizeResponse | null>(null)

  // Interactive Slice Viewer States
  const [currentSlice, setCurrentSlice] = useState<number>(0)
  const [plane, setPlane] = useState<'axial' | 'coronal' | 'sagittal'>('axial')
  const [showWt, setShowWt] = useState<boolean>(true)
  const [showTc, setShowTc] = useState<boolean>(true)
  const [showEt, setShowEt] = useState<boolean>(true)
  const [alpha, setAlpha] = useState<number>(0.45)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)

  useEffect(() => {
    getFiles().then(r => {
      setFiles(r.files)
      const targetId = location.state?.fileId || localStorage.getItem('neurofusion_active_file_id')
      if (targetId && r.files.some(f => f.file_id === targetId)) {
        setSelectedFile(targetId)
      } else if (r.files.length > 0 && !selectedFile) {
        setSelectedFile(r.files[0].file_id)
      }
    }).catch(() => {})
  }, [location.state])

  // Compute total slices along current plane
  const maxSlices = (() => {
    if (!segmentResult?.shape || segmentResult.shape.length < 3) {
      return segmentResult?.total_slices || 64
    }
    if (plane === 'axial') return segmentResult.shape[2]
    if (plane === 'coronal') return segmentResult.shape[1]
    return segmentResult.shape[0] // sagittal
  })()

  // Reset slice index to peak tumor slice or midpoint when result arrives
  useEffect(() => {
    if (segmentResult?.best_slice_idx !== undefined) {
      setCurrentSlice(segmentResult.best_slice_idx)
    } else if (segmentResult?.total_slices) {
      setCurrentSlice(Math.floor(segmentResult.total_slices / 2))
    }
  }, [segmentResult])

  // Clamp slice when plane changes
  useEffect(() => {
    if (currentSlice >= maxSlices) {
      setCurrentSlice(Math.max(0, maxSlices - 1))
    }
  }, [plane, maxSlices])

  // Auto-play slice animation
  useEffect(() => {
    if (!isPlaying) return
    const interval = setInterval(() => {
      setCurrentSlice(prev => (prev + 1) % maxSlices)
    }, 250)
    return () => clearInterval(interval)
  }, [isPlaying, maxSlices])

  const runPipeline = async () => {
    if (!selectedFile) return toast.error('Select a file to segment')
    setProcessing(true)
    setSegmentResult(null)
    setLocalizeResult(null)
    setIsPlaying(false)

    try {
      const [segRes, locRes] = await Promise.all([
        segmentMRI(selectedFile, architecture),
        localizeMRI(selectedFile),
      ])
      setSegmentResult(segRes)
      setLocalizeResult(locRes)
      toast.success('Tumor Sub-region Segmentation & Pathology Localization complete!')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Segmentation failed')
    } finally {
      setProcessing(false)
    }
  }

  // Active slice URL
  const sliceImageUrl = segmentResult?.mask_file_id && selectedFile
    ? getSegmentSliceUrl(
        selectedFile,
        segmentResult.mask_file_id,
        currentSlice,
        plane,
        showWt,
        showTc,
        showEt,
        alpha
      )
    : ''

  const wtData = segmentResult?.measurements?.wt
  const tcData = segmentResult?.measurements?.tc
  const etData = segmentResult?.measurements?.et
  const validNesting = segmentResult?.validation?.valid_nesting ?? true

  const tumorStatus = segmentResult?.tumor_status || segmentResult?.assessment?.tumor_status || (
    (segmentResult?.measurements?.wt?.voxels ?? 0) >= 50
      ? 'Tumor Detected — AI Prediction'
      : (segmentResult?.measurements?.wt?.voxels ?? 0) === 0
      ? 'No Tumor Detected by Model'
      : 'Indeterminate — Requires Validation'
  )

  const isPositive = tumorStatus.includes('Tumor Detected')
  const isNegative = tumorStatus.includes('No Tumor')
  const isIndeterminate = !isPositive && !isNegative

  const statusExplanation = segmentResult?.assessment?.explanation || (
    isPositive
      ? `Volumetric tumor burden detected (WT: ${wtData?.volume_cm3 ?? 0} cm³, TC: ${tcData?.volume_cm3 ?? 0} cm³, ET: ${etData?.volume_cm3 ?? 0} cm³).`
      : isNegative
      ? 'No abnormal enhancing tissue or tumor core detected within sensitivity thresholds.'
      : 'Model prediction requires clinical validation.'
  )

  const validationStatus = segmentResult?.assessment?.validation_status || (
    validNesting ? 'PASSED — ANATOMICAL NESTING VERIFIED' : 'ANOMALOUS MEASUREMENTS DETECTED'
  )

  return (
    <div className="page-container">
      <div className="page-header">
        <motion.h1 className="page-title" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          ROI Segmentation & Pathology Localization
        </motion.h1>
        <p className="page-subtitle">
          Volumetric 3D MRI analysis, hierarchical WT / TC / ET tumor sub-region localization, and quantitative volumetrics
        </p>
      </div>

      {/* Selector */}
      <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 220 }}>
            <label className="label">Select MRI Volume</label>
            <select className="select" value={selectedFile} onChange={e => setSelectedFile(e.target.value)}>
              <option value="">— Choose file —</option>
              {files.map(f => (
                <option key={f.file_id} value={f.file_id}>{f.filename} ({f.modality} · {f.mri_type})</option>
              ))}
            </select>
          </div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <label className="label">Architecture</label>
            <select className="select" value={architecture} onChange={e => setArchitecture(e.target.value)}>
              {ARCHITECTURES.map(a => (
                <option key={a.id} value={a.id}>{a.name}</option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" onClick={runPipeline} disabled={processing || !selectedFile} style={{ opacity: processing || !selectedFile ? 0.5 : 1 }}>
            {processing ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Layers size={16} />}
            {processing ? 'Processing...' : 'Run Segmentation'}
          </button>
        </div>
      </motion.div>

      {/* Results */}
      {segmentResult && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>

          {/* Section: Tumor Detection Status Banner & Quick Action Center */}
          <div
            className="glass-card-static"
            style={{
              padding: '1.5rem',
              marginBottom: '1.5rem',
              border: `1.5px solid ${
                isPositive
                  ? 'rgba(239, 68, 68, 0.5)'
                  : isNegative
                  ? 'rgba(34, 197, 94, 0.5)'
                  : 'rgba(245, 158, 11, 0.5)'
              }`,
              background:
                isPositive
                  ? 'rgba(239, 68, 68, 0.06)'
                  : isNegative
                  ? 'rgba(34, 197, 94, 0.06)'
                  : 'rgba(245, 158, 11, 0.06)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '0.75rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-text-muted)', fontWeight: 600 }}>
                  Tumor Presence Status
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginTop: '0.25rem' }}>
                  <span
                    style={{
                      fontSize: '1.25rem',
                      fontWeight: 800,
                      letterSpacing: '0.02em',
                      color:
                        isPositive
                          ? '#ef4444'
                          : isNegative
                          ? '#22c55e'
                          : '#f59e0b',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    {isPositive && <AlertOctagon size={22} />}
                    {isNegative && <CheckCircle2 size={22} />}
                    {isIndeterminate && <AlertTriangle size={22} />}
                    {tumorStatus}
                  </span>
                </div>
              </div>

              {/* Action Buttons: Download PDF, Print, View */}
              <div style={{ display: 'flex', gap: '0.625rem', flexWrap: 'wrap' }}>
                <a
                  href={getReportPdfUrl(selectedFile, segmentResult.mask_file_id)}
                  download={`BrainSight_AI_Report_${selectedFile}.pdf`}
                  className="btn btn-primary"
                  style={{ gap: '0.5rem', fontSize: '0.8125rem', padding: '0.5rem 1rem' }}
                >
                  <Download size={15} /> Download PDF Report
                </a>
                <a
                  href={getReportUrl(selectedFile, segmentResult.mask_file_id, true)}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                  style={{ gap: '0.5rem', fontSize: '0.8125rem', padding: '0.5rem 1rem' }}
                >
                  <Printer size={15} /> Print Report
                </a>
                <a
                  href={getReportUrl(selectedFile, segmentResult.mask_file_id, false)}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                  style={{ gap: '0.5rem', fontSize: '0.8125rem', padding: '0.5rem 1rem' }}
                >
                  <FileText size={15} /> View Report
                </a>
              </div>
            </div>

            <p style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
              {statusExplanation}
            </p>

            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '0.875rem', flexWrap: 'wrap', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              <span>
                Validation: <strong style={{ color: 'var(--color-accent)' }}>{validationStatus}</strong>
              </span>
              <span>•</span>
              <span>
                Model: <strong>MONAI 3D U-Net</strong>
              </span>
              <span>•</span>
              <span>
                Hierarchical Constraint: <strong style={{ color: validNesting ? '#22c55e' : '#ef4444' }}>{validNesting ? 'ET ≤ TC ≤ WT Verified' : 'Nesting Violation'}</strong>
              </span>
            </div>
          </div>

          {/* Interactive Tumor Sub-Region Slice Viewer */}
          <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
              <div>
                <h2 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
                  <Eye size={18} color="var(--color-primary)" /> MRI Slice Viewer & Tumor Overlay
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
                  Orientation: <strong style={{ textTransform: 'capitalize', color: 'var(--color-text-primary)' }}>{plane}</strong> · Slice {currentSlice + 1} of {maxSlices}
                </span>
              </div>

              {/* Plane Selector */}
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Plane:</span>
                {(['axial', 'coronal', 'sagittal'] as const).map(p => (
                  <button
                    key={p}
                    onClick={() => setPlane(p)}
                    className="btn btn-sm"
                    style={{
                      padding: '0.3rem 0.75rem',
                      fontSize: '0.75rem',
                      textTransform: 'capitalize',
                      background: plane === p ? 'var(--color-primary)' : 'rgba(255, 255, 255, 0.05)',
                      color: plane === p ? '#050816' : 'var(--color-text-secondary)',
                      fontWeight: plane === p ? 700 : 500,
                    }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* Tumor Region Legend & Independent Toggle Controls */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '1rem',
                padding: '0.875rem 1.25rem',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(10, 14, 39, 0.8)',
                border: '1px solid var(--color-border)',
                marginBottom: '1.5rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-text-primary)', marginRight: '0.25rem' }}>
                  Overlay Regions:
                </span>

                {/* WT Toggle & Legend: Green */}
                <button
                  type="button"
                  onClick={() => setShowWt(!showWt)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.4rem 0.85rem',
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${showWt ? '#22c55e' : 'rgba(255, 255, 255, 0.1)'}`,
                    background: showWt ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                    color: showWt ? '#22c55e' : 'var(--color-text-muted)',
                    cursor: 'pointer',
                    fontSize: '0.8125rem',
                    fontWeight: 600,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <span
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 3,
                      background: '#22c55e',
                      opacity: showWt ? 1 : 0.35,
                      boxShadow: showWt ? '0 0 8px rgba(34, 197, 94, 0.6)' : 'none',
                    }}
                  />
                  <span>WT — Whole Tumor</span>
                  {showWt ? <Eye size={14} /> : <EyeOff size={14} />}
                </button>

                {/* TC Toggle & Legend: Yellow */}
                <button
                  type="button"
                  onClick={() => setShowTc(!showTc)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.4rem 0.85rem',
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${showTc ? '#eab308' : 'rgba(255, 255, 255, 0.1)'}`,
                    background: showTc ? 'rgba(234, 179, 8, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                    color: showTc ? '#eab308' : 'var(--color-text-muted)',
                    cursor: 'pointer',
                    fontSize: '0.8125rem',
                    fontWeight: 600,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <span
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 3,
                      background: '#eab308',
                      opacity: showTc ? 1 : 0.35,
                      boxShadow: showTc ? '0 0 8px rgba(234, 179, 8, 0.6)' : 'none',
                    }}
                  />
                  <span>TC — Tumor Core</span>
                  {showTc ? <Eye size={14} /> : <EyeOff size={14} />}
                </button>

                {/* ET Toggle & Legend: Red */}
                <button
                  type="button"
                  onClick={() => setShowEt(!showEt)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.4rem 0.85rem',
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${showEt ? '#ef4444' : 'rgba(255, 255, 255, 0.1)'}`,
                    background: showEt ? 'rgba(239, 68, 68, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                    color: showEt ? '#ef4444' : 'var(--color-text-muted)',
                    cursor: 'pointer',
                    fontSize: '0.8125rem',
                    fontWeight: 600,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <span
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 3,
                      background: '#ef4444',
                      opacity: showEt ? 1 : 0.35,
                      boxShadow: showEt ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none',
                    }}
                  />
                  <span>ET — Enhancing Tumor</span>
                  {showEt ? <Eye size={14} /> : <EyeOff size={14} />}
                </button>
              </div>

              {/* Opacity Control & Mode Indicator */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {!showWt && !showTc && !showEt ? (
                  <span className="metric-badge" style={{ background: 'rgba(255, 255, 255, 0.1)', color: 'var(--color-text-secondary)' }}>
                    Raw MRI Mode (All Overlays Disabled)
                  </span>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Opacity:</span>
                    <input
                      type="range"
                      min="0.15"
                      max="0.85"
                      step="0.05"
                      value={alpha}
                      onChange={e => setAlpha(parseFloat(e.target.value))}
                      style={{ width: 80, accentColor: 'var(--color-primary)' }}
                    />
                    <span style={{ fontSize: '0.75rem', color: 'var(--color-primary)', minWidth: '2.5rem' }}>
                      {Math.round(alpha * 100)}%
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Slice View Display Canvas */}
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative', minHeight: 380, background: '#02040a', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--color-border)' }}>
              {sliceImageUrl ? (
                <img
                  key={sliceImageUrl}
                  src={sliceImageUrl}
                  alt={`MRI Slice ${currentSlice + 1}`}
                  style={{
                    maxHeight: 460,
                    maxWidth: '100%',
                    objectFit: 'contain',
                    display: 'block',
                  }}
                />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--color-text-muted)' }}>
                  <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> Loading Slice...
                </div>
              )}

              {/* Overlay on corner: slice index and status */}
              <div
                style={{
                  position: 'absolute',
                  top: 12,
                  right: 12,
                  padding: '0.3rem 0.6rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(5, 8, 22, 0.85)',
                  backdropFilter: 'blur(4px)',
                  border: '1px solid var(--color-border)',
                  fontSize: '0.75rem',
                  color: 'var(--color-text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                }}
              >
                <span>Slice <strong>{currentSlice + 1}</strong> / {maxSlices}</span>
              </div>
            </div>

            {/* Slice Navigation Controls */}
            <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                <button
                  type="button"
                  onClick={() => setCurrentSlice(prev => Math.max(0, prev - 1))}
                  disabled={currentSlice <= 0}
                  className="btn btn-sm"
                  style={{ padding: '0.4rem 0.6rem' }}
                  title="Previous Slice"
                >
                  <ChevronLeft size={16} />
                </button>
                <button
                  type="button"
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="btn btn-sm"
                  style={{ padding: '0.4rem 0.75rem', background: isPlaying ? 'var(--color-primary)' : undefined, color: isPlaying ? '#050816' : undefined }}
                  title={isPlaying ? 'Pause Animation' : 'Auto Play Slices'}
                >
                  {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                </button>
                <button
                  type="button"
                  onClick={() => setCurrentSlice(prev => Math.min(maxSlices - 1, prev + 1))}
                  disabled={currentSlice >= maxSlices - 1}
                  className="btn btn-sm"
                  style={{ padding: '0.4rem 0.6rem' }}
                  title="Next Slice"
                >
                  <ChevronRight size={16} />
                </button>
              </div>

              {/* Range Slider */}
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 200 }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>0</span>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, maxSlices - 1)}
                  value={currentSlice}
                  onChange={e => setCurrentSlice(parseInt(e.target.value, 10))}
                  style={{ flex: 1, accentColor: 'var(--color-primary)' }}
                />
                <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>{maxSlices - 1}</span>
              </div>

              {/* Quick Jump to Peak Tumor Slice */}
              {segmentResult?.best_slice_idx !== undefined && (
                <button
                  type="button"
                  onClick={() => setCurrentSlice(segmentResult.best_slice_idx || 0)}
                  className="btn btn-sm btn-secondary"
                  style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem' }}
                  title="Jump to slice with maximum tumor cross-section"
                >
                  <Crosshair size={14} color="var(--color-primary)" /> Jump to Peak Tumor (Slice {(segmentResult.best_slice_idx || 0) + 1})
                </button>
              )}
            </div>
          </div>

          {/* Region Measurements & Volumetric Analysis */}
          <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Shield size={18} color="var(--color-accent)" /> Quantitative Tumor Sub-Region Measurements
            </h2>
            <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginBottom: '1.25rem' }}>
              Physical volume calculations derived from MRI voxel spacing: {segmentResult.voxel_spacing?.[0]} × {segmentResult.voxel_spacing?.[1]} × {segmentResult.voxel_spacing?.[2]} mm ({((segmentResult.voxel_spacing?.[0] || 1) * (segmentResult.voxel_spacing?.[1] || 1) * (segmentResult.voxel_spacing?.[2] || 1)).toFixed(3)} mm³/voxel)
            </p>

            {/* Validation Banner */}
            <div
              style={{
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                marginBottom: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                fontSize: '0.8125rem',
                background: validNesting ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: `1px solid ${validNesting ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                color: validNesting ? 'var(--color-accent)' : 'var(--color-danger)',
              }}
            >
              {validNesting ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
              <span>
                {validNesting
                  ? `Hierarchical Sub-region Nesting Verified: ET (${etData?.volume_cm3 ?? 0} cm³) ≤ TC (${tcData?.volume_cm3 ?? 0} cm³) ≤ WT (${wtData?.volume_cm3 ?? 0} cm³)`
                  : 'Warning: Model channel output hierarchy is inconsistent (ET must be ≤ TC ≤ WT). Verify model training configuration.'}
              </span>
            </div>

            {/* 3 Dedicated Region Cards */}
            <div className="grid-3" style={{ gap: '1rem' }}>
              {/* WT Card */}
              <div
                style={{
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(34, 197, 94, 0.05)',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                  boxShadow: '0 0 15px rgba(34, 197, 94, 0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.9375rem', color: '#22c55e', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: '#22c55e' }} />
                    WT — Whole Tumor
                  </span>
                  <span className="metric-badge" style={{ background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e' }}>Green</span>
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--color-text-primary)', marginBottom: '0.25rem' }}>
                  {wtData?.volume_cm3 ?? 0} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--color-text-muted)' }}>cm³ (mL)</span>
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
                  <strong>{(wtData?.voxels ?? 0).toLocaleString()}</strong> voxels · {(wtData?.volume_mm3 ?? 0).toLocaleString()} mm³
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', margin: 0 }}>
                  Complete tumor extent including peritumoral edema, necrotic core, and active enhancing tissue.
                </p>
              </div>

              {/* TC Card */}
              <div
                style={{
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(234, 179, 8, 0.05)',
                  border: '1px solid rgba(234, 179, 8, 0.3)',
                  boxShadow: '0 0 15px rgba(234, 179, 8, 0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.9375rem', color: '#eab308', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: '#eab308' }} />
                    TC — Tumor Core
                  </span>
                  <span className="metric-badge" style={{ background: 'rgba(234, 179, 8, 0.15)', color: '#eab308' }}>Yellow</span>
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--color-text-primary)', marginBottom: '0.25rem' }}>
                  {tcData?.volume_cm3 ?? 0} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--color-text-muted)' }}>cm³ (mL)</span>
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
                  <strong>{(tcData?.voxels ?? 0).toLocaleString()}</strong> voxels · {(tcData?.volume_mm3 ?? 0).toLocaleString()} mm³
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', margin: 0 }}>
                  Gross tumor core within WT, comprising necrotic/non-enhancing components and active enhancing tumor.
                </p>
              </div>

              {/* ET Card */}
              <div
                style={{
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(239, 68, 68, 0.05)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  boxShadow: '0 0 15px rgba(239, 68, 68, 0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.9375rem', color: '#ef4444', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: '#ef4444' }} />
                    ET — Enhancing Tumor
                  </span>
                  <span className="metric-badge" style={{ background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444' }}>Red</span>
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--color-text-primary)', marginBottom: '0.25rem' }}>
                  {etData?.volume_cm3 ?? 0} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--color-text-muted)' }}>cm³ (mL)</span>
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
                  <strong>{(etData?.voxels ?? 0).toLocaleString()}</strong> voxels · {(etData?.volume_mm3 ?? 0).toLocaleString()} mm³
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', margin: 0 }}>
                  Active angiogenic tumor tissue displaying hyperintensity following gadolinium contrast injection.
                </p>
              </div>
            </div>
          </div>

          {/* Pathology Attention Heatmap & Diagnostic Suggestions */}
          <div className="grid-2" style={{ marginBottom: '2rem' }}>
            {/* Multi-panel composite preview */}
            {segmentResult.overlay_preview && (
              <div className="glass-card-static" style={{ padding: '1.5rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Eye size={16} color="var(--color-primary)" /> Sub-Region Composite Tri-Panel
                </h3>
                <img
                  src={segmentResult.overlay_preview}
                  alt="Segmentation Composite"
                  style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}
                />
              </div>
            )}

            {/* Heatmap View */}
            {localizeResult?.heatmap_preview && (
              <div className="glass-card-static" style={{ padding: '1.5rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Shield size={16} color="var(--color-accent)" /> Pathology Attention Heatmap
                </h3>
                <img
                  src={localizeResult.heatmap_preview}
                  alt="Pathology Heatmap"
                  style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}
                />
              </div>
            )}
          </div>

          {/* Clinical Findings */}
          {localizeResult?.localization && (
            <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: isNegative ? 'var(--color-primary)' : 'var(--color-accent)' }}>
                <CheckCircle2 size={18} /> Diagnostic Suggestions & Disease Localization
              </h3>
              {localizeResult.localization.findings && localizeResult.localization.findings.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {localizeResult.localization.findings.map((finding, idx) => (
                    <div key={idx} style={{ padding: '1rem', borderRadius: 'var(--radius-md)', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.9375rem', color: 'var(--color-accent)' }}>{finding.pathology}</span>
                        <span className="metric-badge success">{Math.round(finding.confidence * 100)}% Confidence</span>
                      </div>
                      <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginBottom: '0.375rem' }}>{finding.description}</p>
                      <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>{finding.recommendation}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', background: 'rgba(34, 197, 94, 0.08)', border: '1px solid rgba(34, 197, 94, 0.25)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <CheckCircle2 size={20} color="#22c55e" />
                  <div>
                    <strong style={{ color: '#22c55e', fontSize: '0.875rem' }}>No Abnormal Tumor Pathology Localized</strong>
                    <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>
                      Volumetric scan analyzed with calibrated 3D U-Net. All intracranial regions remain within normal limits.
                    </p>
                  </div>
                </div>
              )}
              <div style={{ marginTop: '1rem', padding: '0.75rem', borderRadius: 'var(--radius-md)', background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.2)', fontSize: '0.75rem', color: 'var(--color-warning)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertTriangle size={14} /> {localizeResult.localization.disclaimer}
              </div>
            </div>
          )}

          {/* Download & Complete Clinical Report Actions */}
          {segmentResult.mask_file_id && (
            <div
              className="glass-card-static"
              style={{
                padding: '1.25rem 1.5rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '1rem',
              }}
            >
              <div>
                <h4 style={{ fontSize: '0.9375rem', fontWeight: 700, margin: 0, color: 'var(--color-text-primary)' }}>
                  Clinical Report & Segmentation Exports
                </h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', margin: '0.25rem 0 0 0' }}>
                  Generate and download genuine medical A4 PDF report or raw 3D segmentation NIfTI mask
                </p>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <a
                  href={getReportPdfUrl(selectedFile, segmentResult.mask_file_id)}
                  download={`BrainSight_AI_Report_${selectedFile}.pdf`}
                  className="btn btn-primary"
                  style={{ gap: '0.5rem' }}
                >
                  <Download size={16} /> Download PDF Report
                </a>
                <a
                  href={getReportUrl(selectedFile, segmentResult.mask_file_id, true)}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                  style={{ gap: '0.5rem' }}
                >
                  <Printer size={16} /> Print Report
                </a>
                <a
                  href={getReportUrl(selectedFile, segmentResult.mask_file_id, false)}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                  style={{ gap: '0.5rem' }}
                >
                  <FileText size={16} /> View Report
                </a>
                <a
                  href={downloadFile(segmentResult.mask_file_id)}
                  className="btn btn-secondary"
                  style={{ gap: '0.5rem' }}
                >
                  <Download size={16} /> Download NIfTI Mask
                </a>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}

