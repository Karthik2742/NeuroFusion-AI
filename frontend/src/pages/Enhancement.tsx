import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  Sparkles, Loader2, Download, BarChart3, Clock, Cpu,
  HardDrive, Zap, Eye,
} from 'lucide-react'
import {
  getFiles, enhanceMRI, getEnhancementModels, downloadFile,
  type FileInfo, type EnhanceResponse, type ModelInfo,
} from '../api'

const MODEL_DESCRIPTIONS: Record<string, string> = {
  clahe: 'Contrast Limited Adaptive Histogram Equalization — Fast baseline enhancement',
  restormer: 'Efficient Transformer with Multi-Dconv Head Transposed Attention (CVPR 2022)',
  swinir: 'Swin Transformer for Image Restoration with window-based self-attention (ICCV 2021)',
  mirnet: 'Multi-scale Residual Learning with Dual Attention Units (ECCV 2020)',
}

const MODEL_COLORS: Record<string, string> = {
  clahe: '#10b981',
  restormer: '#00d4ff',
  swinir: '#7c3aed',
  mirnet: '#f59e0b',
}

export default function Enhancement() {
  const location = useLocation()
  const [files, setFiles] = useState<FileInfo[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedFile, setSelectedFile] = useState('')
  const [selectedModel, setSelectedModel] = useState('clahe')
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<EnhanceResponse | null>(null)

  useEffect(() => {
    Promise.allSettled([getFiles(), getEnhancementModels()]).then(([f, m]) => {
      if (f.status === 'fulfilled') {
        setFiles(f.value.files)
        const targetId = location.state?.fileId || localStorage.getItem('neurofusion_active_file_id')
        if (targetId && f.value.files.some(file => file.file_id === targetId)) {
          setSelectedFile(targetId)
        } else if (f.value.files.length > 0 && !selectedFile) {
          setSelectedFile(f.value.files[0].file_id)
        }
      }
      if (m.status === 'fulfilled') setModels(m.value.models)
    })
  }, [location.state])

  const runEnhancement = async () => {
    if (!selectedFile) return toast.error('Select a file')
    setProcessing(true)
    setResult(null)
    try {
      const res = await enhanceMRI(selectedFile, selectedModel)
      setResult(res)
      if (res.output_file_id) {
        localStorage.setItem('neurofusion_active_file_id', res.output_file_id)
      }
      toast.success(`Enhancement complete using ${selectedModel.toUpperCase()}!`)
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Enhancement failed')
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <motion.h1 className="page-title" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          MRI Enhancement
        </motion.h1>
        <p className="page-subtitle">AI-powered MRI quality enhancement with state-of-the-art deep learning models</p>
      </div>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static"
        style={{ padding: '1.5rem', marginBottom: '2rem' }}
      >
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 220 }}>
            <label className="label">Input MRI File</label>
            <select className="select" value={selectedFile} onChange={e => setSelectedFile(e.target.value)}>
              <option value="">— Choose file —</option>
              {files.map(f => (
                <option key={f.file_id} value={f.file_id}>{f.filename} ({f.modality})</option>
              ))}
            </select>
          </div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <label className="label">Enhancement Model</label>
            <select className="select" value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
              {['clahe', 'restormer', 'swinir', 'mirnet'].map(m => (
                <option key={m} value={m}>{m.toUpperCase()}</option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={runEnhancement}
            disabled={processing || !selectedFile}
            style={{ opacity: processing || !selectedFile ? 0.5 : 1 }}
          >
            {processing ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Sparkles size={16} />}
            {processing ? 'Enhancing...' : 'Enhance'}
          </button>
        </div>

        {/* Model Description */}
        <div style={{
          marginTop: '1rem', padding: '0.75rem 1rem',
          background: `${MODEL_COLORS[selectedModel] || '#00d4ff'}10`,
          border: `1px solid ${MODEL_COLORS[selectedModel] || '#00d4ff'}25`,
          borderRadius: 'var(--radius-md)',
        }}>
          <span style={{
            fontSize: '0.8125rem', color: MODEL_COLORS[selectedModel],
          }}>
            ℹ️ {MODEL_DESCRIPTIONS[selectedModel] || selectedModel}
          </span>
        </div>
      </motion.div>

      {/* Model Cards */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        style={{ marginBottom: '2rem' }}
      >
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem' }}>Available Models</h3>
        <div className="grid-4">
          {['clahe', 'restormer', 'swinir', 'mirnet'].map(model => {
            const isSelected = selectedModel === model
            const color = MODEL_COLORS[model]
            const info = models.find(m => m.name === model)
            return (
              <motion.div
                key={model}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => setSelectedModel(model)}
                style={{
                  padding: '1.25rem',
                  borderRadius: 'var(--radius-lg)',
                  background: isSelected ? `${color}10` : 'var(--color-bg-card)',
                  border: `1px solid ${isSelected ? `${color}40` : 'var(--color-border)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  backdropFilter: 'blur(20px)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: isSelected ? color : 'var(--color-text-muted)',
                  }} />
                  <span style={{
                    fontWeight: 700, fontSize: '0.9375rem',
                    fontFamily: 'var(--font-mono)', color: isSelected ? color : 'var(--color-text-primary)',
                  }}>
                    {model.toUpperCase()}
                  </span>
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', lineHeight: 1.5 }}>
                  {MODEL_DESCRIPTIONS[model]?.split('—')[0] || model}
                </p>
                {info && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.6875rem', color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {info.parameters.total > 0 ? `${(info.parameters.total / 1e6).toFixed(1)}M params` : 'Non-parametric'}
                  </div>
                )}
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Performance Stats */}
          <div className="grid-4" style={{ marginBottom: '2rem' }}>
            {[
              { icon: Zap, label: 'Model', value: result.model_used.toUpperCase(), color: MODEL_COLORS[result.model_used] || '#00d4ff' },
              { icon: Clock, label: 'Inference Time', value: `${result.inference_time_sec.toFixed(2)}s`, color: '#f59e0b' },
              { icon: HardDrive, label: 'Peak Memory', value: `${result.peak_memory_gb.toFixed(3)} GB`, color: '#7c3aed' },
              { icon: Cpu, label: 'Device', value: result.device, color: '#10b981' },
            ].map((stat, i) => {
              const Icon = stat.icon
              return (
                <div key={i} className="glass-card stat-card">
                  <div className="stat-icon" style={{ background: `${stat.color}15` }}>
                    <Icon size={18} color={stat.color} />
                  </div>
                  <div className="stat-value" style={{ color: stat.color, fontSize: '1.25rem' }}>{stat.value}</div>
                  <div className="stat-label">{stat.label}</div>
                </div>
              )
            })}
          </div>

          {/* Quality Metrics Table */}
          {Object.keys(result.metrics).length > 0 && (
            <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart3 size={16} color="var(--color-primary)" /> Enhancement Quality Metrics
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.75rem' }}>
                {Object.entries(result.metrics).map(([key, value]) => (
                  <div key={key} style={{
                    padding: '0.875rem',
                    borderRadius: 'var(--radius-md)',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid var(--color-border)',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)', textTransform: 'uppercase', marginBottom: '0.375rem', letterSpacing: '0.05em' }}>
                      {key}
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--color-primary)' }}>
                      {typeof value === 'number' ? (value > 100 ? value.toFixed(2) : value.toFixed(4)) : value}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Comparison View */}
          {result.comparison_view && (
            <div className="glass-card-static" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Eye size={16} color="var(--color-accent)" /> Before vs After — Difference Map
              </h3>
              <img
                src={result.comparison_view}
                alt="Enhancement comparison"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border)' }}
              />
            </div>
          )}

          {/* Download */}
          {result.output_file_id && (
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <a href={downloadFile(result.output_file_id)} className="btn btn-primary">
                <Download size={16} /> Download Enhanced MRI
              </a>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
