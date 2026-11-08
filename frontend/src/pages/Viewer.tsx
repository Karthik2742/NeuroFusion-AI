import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import { Box, RotateCcw, Layers, Eye, RefreshCw } from 'lucide-react'
import { getFiles, downloadFile, type FileInfo } from '../api'
import { useTheme } from '../context/ThemeContext'
import toast from 'react-hot-toast'

export default function Viewer() {
  const location = useLocation()
  const { theme } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [niivueInstance, setNiivueInstance] = useState<any>(null)
  const [loaded, setLoaded] = useState(false)
  const [viewMode, setViewMode] = useState<string>('multiplanar')

  const [files, setFiles] = useState<FileInfo[]>([])
  const [selectedFile, setSelectedFile] = useState<string>('')
  const [loadingVolume, setLoadingVolume] = useState(false)

  // Initialize NiiVue WebGL Engine
  useEffect(() => {
    let isMounted = true

    const initNiiVue = async () => {
      try {
        const { Niivue } = await import('@niivue/niivue')
        if (!isMounted) return

        if (canvasRef.current) {
          const nv = new Niivue({
            backColor: theme === 'dark' ? [0.03, 0.05, 0.12, 1] : [0.95, 0.96, 0.98, 1],
            crosshairColor: [0, 0.83, 1, 0.6],
            show3Dcrosshair: true,
            isColorbar: true,
            loadingText: 'Loading NIfTI 3D Volume...',
          })

          nv.attachToCanvas(canvasRef.current)
          setNiivueInstance(nv)
          setLoaded(true)
        }
      } catch (err: any) {
        console.warn('NiiVue WebGL initialization fallback:', err)
        if (isMounted) {
          setLoaded(true)
        }
      }
    }

    initNiiVue()

    return () => {
      isMounted = false
    }
  }, [])

  // Sync background color with dark/light theme
  useEffect(() => {
    if (niivueInstance && loaded) {
      try {
        const backColor = theme === 'dark' ? [0.03, 0.05, 0.12, 1] : [0.95, 0.96, 0.98, 1]
        niivueInstance.setBackColor(backColor)
      } catch (e) {
        console.warn('Could not set NiiVue backColor', e)
      }
    }
  }, [theme, niivueInstance, loaded])

  // Load files and auto-select active file
  useEffect(() => {
    getFiles()
      .then((r) => {
        setFiles(r.files)
        const targetId = location.state?.fileId || localStorage.getItem('neurofusion_active_file_id')
        if (targetId && r.files.some((f) => f.file_id === targetId)) {
          setSelectedFile(targetId)
        } else if (r.files.length > 0 && !selectedFile) {
          setSelectedFile(r.files[0].file_id)
        }
      })
      .catch(() => {})
  }, [location.state])

  // Auto-load volume when niivueInstance and selectedFile are ready
  useEffect(() => {
    if (niivueInstance && selectedFile && loaded) {
      loadServerFile(selectedFile)
    }
  }, [niivueInstance, selectedFile, loaded])

  const loadServerFile = async (fileId: string) => {
    if (!fileId) return
    setSelectedFile(fileId)
    setLoadingVolume(true)

    try {
      const fileUrl = downloadFile(fileId)
      const targetFile = files.find((f) => f.file_id === fileId)
      const volumes = [{ url: fileUrl, name: targetFile?.filename || fileId }]

      if (niivueInstance) {
        await niivueInstance.loadVolumes(volumes)
        toast.success(`Rendered ${targetFile?.filename || 'MRI volume'} in 3D viewer`)
      }
    } catch (err) {
      console.error('Failed to load server volume in NiiVue:', err)
      toast.error('Could not load volume into 3D viewer')
    } finally {
      setLoadingVolume(false)
    }
  }

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoadingVolume(true)

    try {
      if (niivueInstance) {
        const volumes = [{ url: URL.createObjectURL(file), name: file.name }]
        await niivueInstance.loadVolumes(volumes)
        toast.success(`Loaded local volume ${file.name}`)
      }
    } catch (err) {
      console.error('Load failed:', err)
      toast.error('Failed to load local NIfTI file')
    } finally {
      setLoadingVolume(false)
    }
  }

  const setSliceType = (mode: string) => {
    setViewMode(mode)
    if (!niivueInstance) return

    const sliceTypes: Record<string, number> = {
      axial: 0,
      coronal: 1,
      sagittal: 2,
      multiplanar: 3,
      render: 4,
    }
    try {
      niivueInstance.setSliceType(sliceTypes[mode] ?? 3)
    } catch (e) {
      console.warn('Set slice type error', e)
    }
  }

  const viewOptions = [
    { key: 'axial', label: 'Axial' },
    { key: 'coronal', label: 'Coronal' },
    { key: 'sagittal', label: 'Sagittal' },
    { key: 'multiplanar', label: 'Multi-Planar' },
    { key: 'render', label: '3D Render' },
  ]

  return (
    <div className="page-container">
      <div className="page-header">
        <motion.h1 className="page-title" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          3D Volume Viewer
        </motion.h1>
        <p className="page-subtitle">Interactive WebGL NIfTI volume rendering with multi-planar views</p>
      </div>

      {/* Toolbar */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card-static"
        style={{
          padding: '1rem 1.25rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          flexWrap: 'wrap',
        }}
      >
        {/* Server File Selector */}
        <div style={{ flex: 1, minWidth: 220, maxWidth: 360 }}>
          <select
            className="select"
            value={selectedFile}
            onChange={(e) => loadServerFile(e.target.value)}
            style={{ padding: '0.45rem 0.75rem', fontSize: '0.8125rem' }}
          >
            <option value="">— Load Uploaded Server Volume —</option>
            {files.map((f) => (
              <option key={f.file_id} value={f.file_id}>
                {f.filename} ({f.modality} · {f.mri_type})
              </option>
            ))}
          </select>
        </div>

        <div style={{ width: 1, height: 28, background: 'var(--color-border)' }} />

        {/* Local File Picker */}
        <div>
          <label
            className="btn btn-secondary"
            style={{ cursor: 'pointer', padding: '0.45rem 0.875rem', fontSize: '0.8125rem' }}
          >
            <Box size={15} /> Open Local .nii/.nii.gz
            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
          </label>
        </div>

        <div style={{ width: 1, height: 28, background: 'var(--color-border)' }} />

        {/* View Mode Buttons */}
        {viewOptions.map((opt) => (
          <button
            key={opt.key}
            className={viewMode === opt.key ? 'btn btn-primary' : 'btn btn-ghost'}
            onClick={() => setSliceType(opt.key)}
            style={{ padding: '0.375rem 0.75rem', fontSize: '0.8125rem' }}
          >
            {opt.label}
          </button>
        ))}

        <div style={{ width: 1, height: 28, background: 'var(--color-border)' }} />

        <button
          className="btn btn-ghost btn-icon"
          title="Reset View"
          onClick={() => {
            try {
              niivueInstance?.setDefaults?.()
            } catch (e) {}
          }}
        >
          <RotateCcw size={16} />
        </button>
      </motion.div>

      {/* Main WebGL Viewer Container */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card-static"
        style={{
          padding: '0.5rem',
          position: 'relative',
          minHeight: 600,
          background: theme === 'dark' ? '#03050f' : '#f1f5f9',
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
        }}
      >
        <canvas
          ref={canvasRef}
          width={1000}
          height={600}
          style={{
            width: '100%',
            height: '600px',
            borderRadius: 'var(--radius-md)',
            display: 'block',
            background: theme === 'dark' ? '#050816' : '#f8fafc',
          }}
        />

        {/* Loading / Status Overlay */}
        {(loadingVolume || !loaded) && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: theme === 'dark' ? 'rgba(5, 8, 22, 0.85)' : 'rgba(255, 255, 255, 0.85)',
              backdropFilter: 'blur(8px)',
              borderRadius: 'var(--radius-lg)',
              zIndex: 10,
            }}
          >
            <RefreshCw size={36} color="var(--color-primary)" style={{ animation: 'spin 1s linear infinite', marginBottom: '1rem' }} />
            <p style={{ color: 'var(--color-text-primary)', fontWeight: 600, fontSize: '0.9375rem' }}>
              {loadingVolume ? 'Rendering 3D NIfTI Volume...' : 'Initializing 3D Viewer Engine...'}
            </p>
          </div>
        )}

        {/* Prompt Overlay when no file is loaded */}
        {!loadingVolume && loaded && files.length === 0 && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: theme === 'dark' ? 'rgba(5, 8, 22, 0.7)' : 'rgba(255, 255, 255, 0.7)',
              borderRadius: 'var(--radius-lg)',
              pointerEvents: 'none',
            }}
          >
            <Eye size={48} color="var(--color-text-muted)" style={{ opacity: 0.4, marginBottom: '1rem' }} />
            <p style={{ color: 'var(--color-text-primary)', fontWeight: 600, fontSize: '1rem', marginBottom: '0.5rem' }}>
              No MRI Volume Loaded
            </p>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.8125rem' }}>
              Upload an MRI file on the Upload page or select a local file to view in 3D
            </p>
          </div>
        )}
      </motion.div>

      {/* Controls & Instructions */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="glass-card-static"
        style={{ padding: '1.25rem', marginTop: '1rem' }}
      >
        <h3
          style={{
            fontSize: '0.9375rem',
            fontWeight: 700,
            marginBottom: '0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Layers size={16} color="var(--color-primary)" /> WebGL 3D Controls & Navigation
        </h3>
        <div className="grid-4" style={{ gap: '0.75rem' }}>
          {[
            { key: 'Scroll Wheel', desc: 'Navigate through 3D slices' },
            { key: 'Right Click + Drag', desc: 'Adjust Window Level / Brightness / Contrast' },
            { key: 'Ctrl + Scroll', desc: 'Zoom in & out of volume' },
            { key: 'Middle Click + Drag', desc: 'Pan volume viewport' },
          ].map((item) => (
            <div key={item.key} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span
                style={{
                  padding: '0.125rem 0.5rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--color-primary-glow)',
                  border: '1px solid var(--color-border-hover)',
                  fontSize: '0.6875rem',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-primary)',
                }}
              >
                {item.key}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>{item.desc}</span>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
