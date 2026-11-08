import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Settings as SettingsIcon, Cpu, Server, RefreshCw, Sun, Moon } from 'lucide-react'
import { getGPUStatus, type GPUStatus } from '../api'
import { useTheme } from '../context/ThemeContext'
import toast from 'react-hot-toast'

export default function Settings() {
  const { theme, setTheme } = useTheme()
  const [gpuStatus, setGpuStatus] = useState<GPUStatus | null>(null)
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [loading, setLoading] = useState(false)

  const refreshGpu = async () => {
    setLoading(true)
    try {
      const gpu = await getGPUStatus()
      setGpuStatus(gpu)
      toast.success('GPU status refreshed')
    } catch {
      toast.error('Could not connect to backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { refreshGpu() }, [])

  return (
    <div className="page-container">
      <div className="page-header">
        <motion.h1 className="page-title" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          Settings
        </motion.h1>
        <p className="page-subtitle">Configure theme appearance, backend connection, GPU, and platform preferences</p>
      </div>

      <div className="grid-2">
        {/* Appearance Settings */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card-static"
          style={{ padding: '1.5rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {theme === 'dark' ? <Moon size={18} color="var(--color-secondary)" /> : <Sun size={18} color="var(--color-warning)" />} Appearance & Theme
          </h3>
          <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)', marginBottom: '1.25rem' }}>
            Switch between dark mode and light mode for optimal visual clarity in clinical or daylight environments.
          </p>

          <div style={{ display: 'flex', gap: '1rem' }}>
            <div
              onClick={() => setTheme('dark')}
              style={{
                flex: 1,
                padding: '1rem',
                borderRadius: 'var(--radius-md)',
                background: theme === 'dark' ? 'var(--color-primary-glow)' : 'var(--color-bg-card)',
                border: `2px solid ${theme === 'dark' ? 'var(--color-primary)' : 'var(--color-border)'}`,
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 0.2s',
              }}
            >
              <Moon size={24} color={theme === 'dark' ? 'var(--color-primary)' : 'var(--color-text-muted)'} style={{ marginBottom: '0.5rem' }} />
              <div style={{ fontWeight: 700, fontSize: '0.875rem' }}>Dark Theme</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Deep neon medical UI</div>
            </div>

            <div
              onClick={() => setTheme('light')}
              style={{
                flex: 1,
                padding: '1rem',
                borderRadius: 'var(--radius-md)',
                background: theme === 'light' ? 'var(--color-primary-glow)' : 'var(--color-bg-card)',
                border: `2px solid ${theme === 'light' ? 'var(--color-primary)' : 'var(--color-border)'}`,
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 0.2s',
              }}
            >
              <Sun size={24} color={theme === 'light' ? '#f59e0b' : 'var(--color-text-muted)'} style={{ marginBottom: '0.5rem' }} />
              <div style={{ fontWeight: 700, fontSize: '0.875rem' }}>Light Theme</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Clean high-contrast UI</div>
            </div>
          </div>
        </motion.div>

        {/* Backend Connection */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card-static"
          style={{ padding: '1.5rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Server size={18} color="var(--color-primary)" /> Backend Connection
          </h3>
          <div style={{ marginBottom: '1rem' }}>
            <label className="label">API URL</label>
            <input
              className="input"
              value={apiUrl}
              onChange={e => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>
          <button className="btn btn-secondary" onClick={refreshGpu}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Test Connection
          </button>
        </motion.div>

        {/* GPU Configuration */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card-static"
          style={{ padding: '1.5rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={18} color="var(--color-accent)" /> GPU Status
          </h3>
          {gpuStatus ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>Available</span>
                <span className={`metric-badge ${gpuStatus.is_available ? 'success' : 'warning'}`}>
                  {gpuStatus.is_available ? 'Yes' : 'No'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>Device</span>
                <span style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)' }}>{gpuStatus.device_name}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>CUDA</span>
                <span style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)' }}>{gpuStatus.cuda_version}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>Total Memory</span>
                <span style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)' }}>{gpuStatus.total_memory_gb} GB</span>
              </div>
              <div style={{ marginTop: '0.5rem' }}>
                <div className="progress-bar" style={{ height: 6 }}>
                  <div className="progress-fill" style={{ width: `${gpuStatus.utilization_percent}%` }} />
                </div>
                <p style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)', marginTop: '0.25rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                  {gpuStatus.utilization_percent}% utilized
                </p>
              </div>
            </div>
          ) : (
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>
              {loading ? 'Checking...' : 'Could not connect to backend'}
            </p>
          )}
        </motion.div>

        {/* Platform Info */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card-static"
          style={{ padding: '1.5rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <SettingsIcon size={18} color="var(--color-warning)" /> Platform Information
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
            {[
              { label: 'Platform', value: 'NeuroFusion AI v1.0.0' },
              { label: 'Backend', value: 'FastAPI + PyTorch + MONAI' },
              { label: 'Frontend', value: 'React + TypeScript + Vite' },
              { label: 'Viewer', value: 'NiiVue (WebGL)' },
              { label: 'Enhancement Models', value: 'CLAHE, Restormer, SwinIR, MIRNet' },
              { label: 'Dataset', value: 'BraTS 2020 (Kaggle)' },
            ].map((item, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>{item.label}</span>
                <span style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}>{item.value}</span>
              </div>
            ))}
          </div>
          <div style={{
            marginTop: '1.25rem', padding: '0.75rem', borderRadius: 'var(--radius-md)',
            background: 'rgba(245, 158, 11, 0.06)', border: '1px solid rgba(245, 158, 11, 0.15)',
            fontSize: '0.75rem', color: 'var(--color-warning)',
          }}>
            ⚠️ This platform is for research and demonstration purposes only. Not intended for clinical diagnosis.
          </div>
        </motion.div>
      </div>
    </div>
  )
}
