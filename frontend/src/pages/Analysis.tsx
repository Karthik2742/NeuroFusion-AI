import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  BarChart3, Loader2, Brain, Layers, Ruler,
  Volume2, TrendingUp,
} from 'lucide-react'
import { getFiles, analyzeMRI, type FileInfo, type AnalysisResponse } from '../api'

export default function Analysis() {
  const location = useLocation()
  const [files, setFiles] = useState<FileInfo[]>([])
  const [selectedFile, setSelectedFile] = useState<string>('')
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResponse | null>(null)

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

  const runAnalysis = async () => {
    if (!selectedFile) {
      toast.error('Select a file to analyze')
      return
    }
    setAnalyzing(true)
    setResult(null)
    try {
      const res = await analyzeMRI(selectedFile)
      setResult(res)
      toast.success('Analysis complete!')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <motion.h1 className="page-title" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          Dataset Analysis
        </motion.h1>
        <p className="page-subtitle">Comprehensive MRI volume analysis — statistics, quality metrics & visualizations</p>
      </div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static"
        style={{ padding: '1.5rem', marginBottom: '2rem' }}
      >
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 250 }}>
            <label className="label">Select MRI File</label>
            <select
              className="select"
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
            >
              <option value="">— Choose an uploaded file —</option>
              {files.map(f => (
                <option key={f.file_id} value={f.file_id}>
                  {f.filename} ({f.modality} · {f.mri_type})
                </option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={runAnalysis}
            disabled={analyzing || !selectedFile}
            style={{ opacity: analyzing || !selectedFile ? 0.5 : 1 }}
          >
            {analyzing ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <BarChart3 size={16} />}
            {analyzing ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
        {files.length === 0 && (
          <p style={{ marginTop: '0.75rem', fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>
            No files uploaded yet. Go to Upload page first.
          </p>
        )}
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Detection Result */}
          <div className="grid-3" style={{ marginBottom: '2rem' }}>
            <div className="glass-card stat-card">
              <div className="stat-icon" style={{ background: 'var(--color-primary-glow)' }}>
                <Brain size={20} color="var(--color-primary)" />
              </div>
              <div className="stat-value" style={{ color: 'var(--color-primary)' }}>{result.mri_type.toUpperCase()}</div>
              <div className="stat-label">Detected MRI Type</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-icon" style={{ background: 'var(--color-secondary-glow)' }}>
                <Layers size={20} color="var(--color-secondary)" />
              </div>
              <div className="stat-value" style={{ color: 'var(--color-secondary)', fontSize: '1.5rem' }}>{result.modality}</div>
              <div className="stat-label">Modality</div>
            </div>
            <div className="glass-card stat-card">
              <div className="stat-icon" style={{ background: 'var(--color-accent-glow)' }}>
                <Ruler size={20} color="var(--color-accent)" />
              </div>
              <div className="stat-value" style={{ color: 'var(--color-accent)', fontSize: '1.25rem' }}>
                {result.spatial_stats?.shape?.slice(0, 3).join(' × ')}
              </div>
              <div className="stat-label">Volume Shape</div>
            </div>
          </div>

          {/* Spatial & Quality Stats */}
          <div className="grid-2" style={{ marginBottom: '2rem' }}>
            {/* Spatial */}
            <div className="glass-card-static" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Ruler size={16} color="var(--color-primary)" /> Spatial Statistics
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
                {[
                  { label: 'Voxel Spacing', value: result.spatial_stats?.voxel_spacing?.map((v: number) => v.toFixed(2)).join(' × ') + ' mm' },
                  { label: 'Orientation', value: result.spatial_stats?.orientation },
                  { label: 'Physical Size', value: result.spatial_stats?.physical_size_mm?.map((v: number) => v.toFixed(1)).join(' × ') + ' mm' },
                  { label: 'Total Voxels', value: result.spatial_stats?.total_voxels?.toLocaleString() },
                  { label: 'Foreground Volume', value: result.spatial_stats?.volume_ml + ' mL' },
                ].map((item, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>{item.label}</span>
                    <span style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quality */}
            <div className="glass-card-static" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <TrendingUp size={16} color="var(--color-accent)" /> Quality Metrics
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
                {Object.entries(result.quality_metrics || {}).map(([key, value]) => (
                  <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>{key}</span>
                    <span className="metric-badge success">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Intensity Stats */}
          <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Volume2 size={16} color="var(--color-warning)" /> Intensity Statistics
            </h3>
            <div className="grid-2">
              {['global', 'foreground'].map(category => (
                <div key={category}>
                  <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-primary)', marginBottom: '0.5rem', textTransform: 'capitalize' }}>
                    {category}
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
                    {Object.entries(result.intensity_stats?.[category] || {}).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>{k}</span>
                        <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}>
                          {typeof v === 'number' ? v.toFixed(2) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Charts */}
          {result.charts && Object.keys(result.charts).length > 0 && (
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '1rem' }}>Visualizations</h3>
              <div className="grid-2">
                {Object.entries(result.charts).map(([key, base64]) => (
                  <div key={key} className="glass-card-static" style={{ padding: '1rem', overflow: 'hidden' }}>
                    <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-text-secondary)', marginBottom: '0.75rem', textTransform: 'capitalize' }}>
                      {key.replace(/_/g, ' ')}
                    </h4>
                    <img
                      src={base64}
                      alt={key}
                      style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
