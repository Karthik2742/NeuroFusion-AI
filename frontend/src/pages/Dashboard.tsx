import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  Upload, Brain, Sparkles, BarChart3, Cpu, HardDrive,
  Activity, Clock, FileImage, Sliders, Award,
} from 'lucide-react'
import { getGPUStatus, getFiles, type GPUStatus, type FileInfo } from '../api'

export default function Dashboard() {
  const navigate = useNavigate()
  const [gpuStatus, setGpuStatus] = useState<GPUStatus | null>(null)
  const [files, setFiles] = useState<FileInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [gpu, filesRes] = await Promise.allSettled([
          getGPUStatus(),
          getFiles(),
        ])
        if (gpu.status === 'fulfilled') setGpuStatus(gpu.value)
        if (filesRes.status === 'fulfilled') setFiles(filesRes.value.files)
      } catch (e) {
        console.warn('API not available yet')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const quickActions = [
    { icon: Upload, label: 'Upload MRI', desc: 'Upload .nii/.nii.gz files', path: '/upload', color: '#00d4ff' },
    { icon: BarChart3, label: 'Analyze Volume', desc: 'Auto-detect & statistics', path: '/analysis', color: '#7c3aed' },
    { icon: Sliders, label: 'Preprocessing', desc: 'N4 bias & normalization', path: '/preprocessing', color: '#ec4899' },
    { icon: Sparkles, label: 'Enhance MRI', desc: 'AI-powered enhancement', path: '/enhancement', color: '#10b981' },
    { icon: Award, label: 'Evaluation', desc: 'PSNR, SSIM & quality', path: '/evaluation', color: '#3b82f6' },
    { icon: Brain, label: '3D Viewer', desc: 'Volume visualization', path: '/viewer', color: '#f59e0b' },
  ]

  const statCards = [
    {
      icon: FileImage, label: 'Uploaded Files', value: files.length.toString(),
      color: '#00d4ff', bg: 'var(--color-primary-glow)',
    },
    {
      icon: Cpu, label: 'GPU Status',
      value: gpuStatus?.is_available ? 'Active' : 'CPU Only',
      color: gpuStatus?.is_available ? '#10b981' : '#f59e0b',
      bg: gpuStatus?.is_available ? 'var(--color-accent-glow)' : 'rgba(245, 158, 11, 0.1)',
    },
    {
      icon: HardDrive, label: 'GPU Memory',
      value: gpuStatus?.is_available ? `${gpuStatus.free_memory_gb} GB Free` : 'N/A',
      color: '#7c3aed', bg: 'var(--color-secondary-glow)',
    },
    {
      icon: Activity, label: 'GPU Utilization',
      value: gpuStatus?.is_available ? `${gpuStatus.utilization_percent}%` : '0%',
      color: '#ec4899', bg: 'rgba(236, 72, 153, 0.1)',
    },
  ]

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <motion.h1
          className="page-title"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          Dashboard
        </motion.h1>
        <p className="page-subtitle">Welcome to NeuroFusion AI — Your medical imaging command center</p>
      </div>

      {/* Stat Cards */}
      <motion.div
        className="grid-4"
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        style={{ marginBottom: '2rem' }}
      >
        {statCards.map((stat, i) => {
          const Icon = stat.icon
          return (
            <div key={i} className="glass-card stat-card">
              <div className="stat-icon" style={{ background: stat.bg }}>
                <Icon size={20} color={stat.color} />
              </div>
              <div className="stat-value" style={{ color: stat.color }}>{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          )
        })}
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        style={{ marginBottom: '2rem' }}
      >
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--color-text-primary)' }}>
          Quick Actions
        </h2>
        <div className="grid-3">
          {quickActions.map((action, i) => {
            const Icon = action.icon
            return (
              <motion.div
                key={i}
                whileHover={{ scale: 1.03, y: -3 }}
                whileTap={{ scale: 0.97 }}
                className="glass-card"
                style={{ padding: '1.5rem', cursor: 'pointer', textAlign: 'center' }}
                onClick={() => navigate(action.path)}
              >
                <div style={{
                  width: 56, height: 56, borderRadius: 'var(--radius-lg)',
                  background: `${action.color}12`, border: `1px solid ${action.color}25`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 1rem',
                }}>
                  <Icon size={28} color={action.color} />
                </div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.25rem' }}>{action.label}</h3>
                <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>{action.desc}</p>
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* GPU Details & Recent Files */}
      <div className="grid-2">
        {/* GPU Card */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="glass-card-static"
          style={{ padding: '1.5rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={18} color="var(--color-primary)" /> GPU Information
          </h3>
          {gpuStatus?.is_available ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {[
                { label: 'Device', value: gpuStatus.device_name },
                { label: 'CUDA Version', value: gpuStatus.cuda_version },
                { label: 'Total Memory', value: `${gpuStatus.total_memory_gb} GB` },
                { label: 'Allocated', value: `${gpuStatus.allocated_memory_gb} GB` },
                { label: 'Free', value: `${gpuStatus.free_memory_gb} GB` },
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>{item.label}</span>
                  <span style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}>{item.value}</span>
                </div>
              ))}
              <div style={{ marginTop: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Memory Usage</span>
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--color-primary)' }}>{gpuStatus.utilization_percent}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${gpuStatus.utilization_percent}%` }} />
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem 0' }}>
              <Cpu size={40} color="var(--color-text-muted)" style={{ marginBottom: '0.75rem', opacity: 0.3 }} />
              <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>
                {loading ? 'Checking GPU status...' : 'No GPU detected — Running in CPU mode'}
              </p>
              <p style={{ color: 'var(--color-text-muted)', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                Start the backend server for GPU information
              </p>
            </div>
          )}
        </motion.div>

        {/* Recent Files */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="glass-card-static"
          style={{ padding: '1.5rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Clock size={18} color="var(--color-secondary)" /> Recent Files
          </h3>
          {files.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {files.slice(0, 5).map((file, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: '0.75rem',
                  padding: '0.625rem', borderRadius: 'var(--radius-md)',
                  background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--color-border)',
                }}>
                  <FileImage size={16} color="var(--color-primary)" />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {file.filename}
                    </div>
                    <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)' }}>
                      {file.modality} · {file.mri_type} · {file.shape.join('×')}
                    </div>
                  </div>
                  <span className="metric-badge info">{file.modality}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem 0' }}>
              <Upload size={40} color="var(--color-text-muted)" style={{ marginBottom: '0.75rem', opacity: 0.3 }} />
              <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>No files uploaded yet</p>
              <button className="btn btn-primary" style={{ marginTop: '0.75rem' }} onClick={() => navigate('/upload')}>
                Upload First MRI
              </button>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
