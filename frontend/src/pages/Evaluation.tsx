import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  BarChart3, Loader2, CheckCircle2,
  FileImage, Award, Zap
} from 'lucide-react'
import { getFiles, evaluateMRI, type FileInfo } from '../api'

export default function Evaluation() {
  const location = useLocation()
  const [files, setFiles] = useState<FileInfo[]>([])
  const [originalId, setOriginalId] = useState('')
  const [enhancedId, setEnhancedId] = useState('')
  const [evaluating, setEvaluating] = useState(false)
  const [result, setResult] = useState<any>(null)

  useEffect(() => {
    getFiles().then((r) => {
      setFiles(r.files)
      const targetId = location.state?.fileId || localStorage.getItem('neurofusion_active_file_id')
      if (targetId && r.files.some((f) => f.file_id === targetId)) {
        setOriginalId(targetId)
        const enhancedCandidate = r.files.find((f) => f.file_id !== targetId)
        if (enhancedCandidate) setEnhancedId(enhancedCandidate.file_id)
      } else if (r.files.length > 0) {
        setOriginalId(r.files[0].file_id)
        if (r.files.length > 1) {
          setEnhancedId(r.files[1].file_id)
        }
      }
    }).catch(() => {})
  }, [location.state])

  const runEvaluation = async () => {
    if (!originalId || !enhancedId) {
      return toast.error('Select both original and enhanced files to evaluate')
    }
    if (originalId === enhancedId) {
      return toast.error('Select two different files for comparison')
    }

    setEvaluating(true)
    setResult(null)

    try {
      const res = await evaluateMRI(originalId, enhancedId)
      setResult(res)
      toast.success('Evaluation metrics computed successfully!')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Evaluation failed')
    } finally {
      setEvaluating(false)
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
          Quantitative MRI Quality Evaluation
        </motion.h1>
        <p className="page-subtitle">
          Compute PSNR, SSIM, MSE, MAE, Sharpness, Contrast, and 15+ quantitative metrics comparing original vs enhanced MRI
        </p>
      </div>

      {/* Selectors */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static"
        style={{ padding: '1.5rem', marginBottom: '2rem' }}
      >
        <div className="grid-2" style={{ gap: '1.5rem', marginBottom: '1.5rem' }}>
          <div>
            <label className="label">1. Original Baseline MRI</label>
            <select
              className="select"
              value={originalId}
              onChange={(e) => setOriginalId(e.target.value)}
            >
              <option value="">— Select original file —</option>
              {files.map((f) => (
                <option key={f.file_id} value={f.file_id}>
                  {f.filename} ({f.modality} · {f.mri_type})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label">2. Enhanced / Processed MRI</label>
            <select
              className="select"
              value={enhancedId}
              onChange={(e) => setEnhancedId(e.target.value)}
            >
              <option value="">— Select enhanced file —</option>
              {files.map((f) => (
                <option key={f.file_id} value={f.file_id}>
                  {f.filename} ({f.modality} · {f.mri_type})
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={runEvaluation}
          disabled={evaluating || !originalId || !enhancedId}
          style={{ opacity: evaluating || !originalId || !enhancedId ? 0.5 : 1 }}
        >
          {evaluating ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <BarChart3 size={16} />}
          {evaluating ? 'Computing Evaluation Metrics...' : 'Evaluate Quality Metrics'}
        </button>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="grid-3" style={{ marginBottom: '2rem' }}>
            {/* PSNR */}
            {result.enhancement_metrics?.psnr !== undefined && (
              <div className="glass-card stat-card">
                <div className="stat-icon" style={{ background: 'var(--color-primary-glow)' }}>
                  <Zap size={20} color="var(--color-primary)" />
                </div>
                <div className="stat-value" style={{ color: 'var(--color-primary)' }}>
                  {result.enhancement_metrics.psnr.toFixed(2)} dB
                </div>
                <div className="stat-label">PSNR (Peak Signal-to-Noise)</div>
              </div>
            )}

            {/* SSIM */}
            {result.enhancement_metrics?.ssim !== undefined && (
              <div className="glass-card stat-card">
                <div className="stat-icon" style={{ background: 'var(--color-accent-glow)' }}>
                  <Award size={20} color="var(--color-accent)" />
                </div>
                <div className="stat-value" style={{ color: 'var(--color-accent)' }}>
                  {result.enhancement_metrics.ssim.toFixed(4)}
                </div>
                <div className="stat-label">SSIM (Structural Similarity)</div>
              </div>
            )}

            {/* MSE */}
            {result.enhancement_metrics?.mse !== undefined && (
              <div className="glass-card stat-card">
                <div className="stat-icon" style={{ background: 'var(--color-secondary-glow)' }}>
                  <CheckCircle2 size={20} color="var(--color-secondary)" />
                </div>
                <div className="stat-value" style={{ color: 'var(--color-secondary)' }}>
                  {result.enhancement_metrics.mse.toFixed(6)}
                </div>
                <div className="stat-label">MSE (Mean Squared Error)</div>
              </div>
            )}
          </div>

          {/* Detailed Quality Metrics Table */}
          {result.enhancement_metrics && (
            <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart3 size={16} color="var(--color-primary)" /> Comprehensive Quality Metrics Breakdown
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
                {Object.entries(result.enhancement_metrics).map(([key, value]) => (
                  <div
                    key={key}
                    style={{
                      padding: '1rem',
                      borderRadius: 'var(--radius-md)',
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid var(--color-border)',
                      textAlign: 'center',
                    }}
                  >
                    <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: '0.375rem', letterSpacing: '0.05em' }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--color-primary)' }}>
                      {typeof value === 'number'
                        ? (value > 100 ? value.toFixed(2) : value.toFixed(4))
                        : String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Evaluation Chart */}
          {result.charts?.metrics_chart && (
            <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <FileImage size={16} color="var(--color-accent)" /> Quality Comparison Chart
              </h3>
              <img
                src={result.charts.metrics_chart}
                alt="Metrics Chart"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}
              />
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
