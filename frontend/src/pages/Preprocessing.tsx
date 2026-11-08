import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  Sliders, Loader2, Download, CheckCircle2,
  Brain, FileText, ArrowRight, RefreshCw, Eye
} from 'lucide-react'
import { getFiles, preprocessMRI, downloadFile, type FileInfo } from '../api'

const PREPROCESSING_STEPS = [
  {
    id: 'bias_correction',
    name: 'N4 Bias Field Correction',
    desc: 'Corrects low-frequency magnetic field inhomogeneity & intensity non-uniformity',
    defaultChecked: true,
  },
  {
    id: 'normalize',
    name: 'Z-Score Intensity Normalization',
    desc: 'Standardizes voxel intensities across foreground tissue (zero mean, unit variance)',
    defaultChecked: true,
  },
  {
    id: 'skull_strip',
    name: 'Skull Stripping',
    desc: 'Isolates brain tissue and removes non-brain cranial structures',
    defaultChecked: false,
  },
  {
    id: 'resample',
    name: 'Isotropic Resampling (1mm³)',
    desc: 'Resamples volume voxels to uniform 1.0mm × 1.0mm × 1.0mm isotropic spacing',
    defaultChecked: false,
  },
  {
    id: 'crop',
    name: 'Crop to Foreground',
    desc: 'Trims non-informative zero-background bounding box for compact memory usage',
    defaultChecked: true,
  },
  {
    id: 'pad',
    name: 'Pad Volume (Multiple of 16)',
    desc: 'Pads dimensions to multiples of 16 for optimal UNet/Transformer tensor shapes',
    defaultChecked: false,
  },
]

export default function Preprocessing() {
  const location = useLocation()
  const [files, setFiles] = useState<FileInfo[]>([])
  const [selectedFile, setSelectedFile] = useState('')
  const [selectedSteps, setSelectedSteps] = useState<string[]>([
    'bias_correction',
    'normalize',
    'crop',
  ])
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<any>(null)

  useEffect(() => {
    getFiles().then((r) => {
      setFiles(r.files)
      const targetId = location.state?.fileId || localStorage.getItem('neurofusion_active_file_id')
      if (targetId && r.files.some((f) => f.file_id === targetId)) {
        setSelectedFile(targetId)
      } else if (r.files.length > 0 && !selectedFile) {
        setSelectedFile(r.files[0].file_id)
      }
    }).catch(() => {})
  }, [location.state])

  const toggleStep = (stepId: string) => {
    setSelectedSteps((prev) =>
      prev.includes(stepId) ? prev.filter((s) => s !== stepId) : [...prev, stepId]
    )
  }

  const runPreprocessing = async () => {
    if (!selectedFile) return toast.error('Select an MRI file to preprocess')
    if (selectedSteps.length === 0) return toast.error('Select at least one preprocessing step')

    setProcessing(true)
    setResult(null)

    try {
      const res = await preprocessMRI(selectedFile, selectedSteps)
      setResult(res)
      toast.success('Preprocessing completed successfully!')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Preprocessing failed')
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <motion.h1
          className="page-title"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          MRI Preprocessing Pipeline
        </motion.h1>
        <p className="page-subtitle">
          Prepare 3D MRI volumes with N4 bias correction, normalization, skull stripping, and resampling
        </p>
      </div>

      {/* File & Pipeline Selector */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static"
        style={{ padding: '1.5rem', marginBottom: '2rem' }}
      >
        <div style={{ marginBottom: '1.5rem' }}>
          <label className="label">Select Uploaded MRI File</label>
          <select
            className="select"
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
          >
            <option value="">— Select an MRI file —</option>
            {files.map((f) => (
              <option key={f.file_id} value={f.file_id}>
                {f.filename} ({f.modality} · {f.mri_type})
              </option>
            ))}
          </select>
          {files.length === 0 && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              No files uploaded yet. Upload a file on the Upload page first.
            </p>
          )}
        </div>

        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sliders size={18} color="var(--color-primary)" /> Select Preprocessing Steps
        </h3>

        <div className="grid-2" style={{ gap: '1rem', marginBottom: '1.5rem' }}>
          {PREPROCESSING_STEPS.map((step) => {
            const isChecked = selectedSteps.includes(step.id)
            return (
              <div
                key={step.id}
                onClick={() => toggleStep(step.id)}
                style={{
                  padding: '1rem',
                  borderRadius: 'var(--radius-md)',
                  background: isChecked ? 'var(--color-primary-glow)' : 'var(--color-bg-card)',
                  border: `1px solid ${isChecked ? 'var(--color-primary)' : 'var(--color-border)'}`,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                  transition: 'all 0.2s',
                }}
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => {}}
                  style={{ marginTop: 3, cursor: 'pointer' }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--color-text-primary)' }}>
                    {step.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                    {step.desc}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-primary"
            onClick={runPreprocessing}
            disabled={processing || !selectedFile}
            style={{ opacity: processing || !selectedFile ? 0.5 : 1 }}
          >
            {processing ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <RefreshCw size={16} />}
            {processing ? 'Processing Pipeline...' : 'Run Preprocessing Pipeline'}
          </button>
        </div>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Summary Cards */}
          <div className="grid-3" style={{ marginBottom: '2rem' }}>
            <div className="glass-card stat-card">
              <div className="stat-icon" style={{ background: 'var(--color-accent-glow)' }}>
                <CheckCircle2 size={20} color="var(--color-accent)" />
              </div>
              <div className="stat-value" style={{ color: 'var(--color-accent)' }}>
                {result.steps_applied?.length || 0}
              </div>
              <div className="stat-label">Steps Applied</div>
            </div>

            <div className="glass-card stat-card">
              <div className="stat-icon" style={{ background: 'var(--color-primary-glow)' }}>
                <Brain size={20} color="var(--color-primary)" />
              </div>
              <div className="stat-value" style={{ color: 'var(--color-primary)', fontSize: '1.25rem' }}>
                {result.original_shape?.join(' × ')}
              </div>
              <div className="stat-label">Original Shape</div>
            </div>

            <div className="glass-card stat-card">
              <div className="stat-icon" style={{ background: 'var(--color-secondary-glow)' }}>
                <ArrowRight size={20} color="var(--color-secondary)" />
              </div>
              <div className="stat-value" style={{ color: 'var(--color-secondary)', fontSize: '1.25rem' }}>
                {result.output_shape?.join(' × ')}
              </div>
              <div className="stat-label">Preprocessed Shape</div>
            </div>
          </div>

          {/* Preview Image */}
          {result.preview && (
            <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Eye size={16} color="var(--color-primary)" /> Preprocessed Volume Multi-planar View
              </h3>
              <img
                src={result.preview}
                alt="Preprocessed Preview"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}
              />
            </div>
          )}

          {/* Applied Step Details */}
          <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileText size={16} color="var(--color-accent)" /> Pipeline Step Details
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {Object.entries(result.step_details || {}).map(([stepKey, details]: [string, any]) => (
                <div
                  key={stepKey}
                  style={{
                    padding: '0.875rem 1rem',
                    borderRadius: 'var(--radius-md)',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid var(--color-border)',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--color-primary)', textTransform: 'capitalize', marginBottom: '0.375rem' }}>
                    {stepKey.replace(/_/g, ' ')}
                  </div>
                  <pre style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(details, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>

          {/* Download Preprocessed Volume */}
          {result.output_file_id && (
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <a href={downloadFile(result.output_file_id)} className="btn btn-primary">
                <Download size={16} /> Download Preprocessed NIfTI Volume
              </a>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
