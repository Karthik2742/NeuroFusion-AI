import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  Upload as UploadIcon, FileImage, Brain, CheckCircle2,
  AlertCircle, Loader2, X, Eye, BarChart3, Sparkles, Sliders, Layers,
} from 'lucide-react'
import { uploadMRI, type UploadResponse } from '../api'

export default function Upload() {
  const navigate = useNavigate()
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    const validExts = ['.nii', '.nii.gz']
    const isValid = validExts.some(ext => file.name.toLowerCase().endsWith(ext))
    if (!isValid) {
      toast.error('Invalid file format. Only .nii and .nii.gz files are accepted.')
      return
    }

    setUploading(true)
    setError(null)
    setResult(null)
    setProgress(0)

    const progressInterval = setInterval(() => {
      setProgress(prev => Math.min(prev + Math.random() * 15, 90))
    }, 200)

    try {
      const response = await uploadMRI(file)
      clearInterval(progressInterval)
      setProgress(100)
      setResult(response)
      localStorage.setItem('neurofusion_active_file_id', response.file_id)
      toast.success(`${file.name} uploaded successfully!`)
    } catch (err: any) {
      clearInterval(progressInterval)
      const msg = err?.response?.data?.detail || 'Upload failed. Is the backend running?'
      setError(msg)
      toast.error(msg)
    } finally {
      setUploading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/gzip': ['.nii.gz'], 'application/octet-stream': ['.nii'] },
    maxFiles: 1,
    disabled: uploading,
  })

  const reset = () => {
    setResult(null)
    setError(null)
    setProgress(0)
    setUploading(false)
  }

  const navigateWithFile = (path: string) => {
    if (result?.file_id) {
      navigate(path, { state: { fileId: result.file_id } })
    } else {
      navigate(path)
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
          Upload MRI
        </motion.h1>
        <p className="page-subtitle">Upload NIfTI Brain MRI volumes for analysis</p>
      </div>

      <div style={{ maxWidth: 800, margin: '0 auto' }}>
        {/* Dropzone */}
        <AnimatePresence mode="wait">
          {!result ? (
            <motion.div
              key="dropzone"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
            >
              <div
                {...getRootProps()}
                style={{
                  padding: '4rem 2rem',
                  borderRadius: 'var(--radius-xl)',
                  border: `2px dashed ${isDragActive ? 'var(--color-primary)' : 'var(--color-border)'}`,
                  background: isDragActive ? 'var(--color-primary-glow)' : 'var(--color-bg-card)',
                  textAlign: 'center',
                  cursor: uploading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.3s',
                  backdropFilter: 'blur(20px)',
                }}
              >
                <input {...getInputProps()} />

                {uploading ? (
                  <div>
                    <Loader2 size={48} color="var(--color-primary)" style={{ animation: 'spin 1s linear infinite', marginBottom: '1rem' }} />
                    <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>Processing MRI Volume...</h3>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1.5rem', fontSize: '0.875rem' }}>
                      Loading, validating, and generating preview
                    </p>
                    <div style={{ maxWidth: 400, margin: '0 auto' }}>
                      <div className="progress-bar" style={{ height: 8 }}>
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                      <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.5rem', fontFamily: 'var(--font-mono)' }}>
                        {Math.round(progress)}%
                      </p>
                    </div>
                  </div>
                ) : (
                  <div>
                    <motion.div
                      animate={isDragActive ? { scale: 1.1, y: -5 } : { scale: 1, y: 0 }}
                      transition={{ type: 'spring', stiffness: 300 }}
                    >
                      <div style={{
                        width: 80, height: 80, borderRadius: 'var(--radius-xl)',
                        background: 'var(--gradient-primary)', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem',
                        boxShadow: '0 8px 32px rgba(0, 212, 255, 0.25)',
                      }}>
                        <UploadIcon size={36} color="white" />
                      </div>
                    </motion.div>

                    <h3 style={{ fontSize: '1.375rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                      {isDragActive ? 'Drop your MRI file here' : 'Drag & Drop MRI Volume'}
                    </h3>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: '1rem', fontSize: '0.9375rem' }}>
                      or click to browse files
                    </p>

                    <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                      {['.nii', '.nii.gz'].map(ext => (
                        <span key={ext} className="metric-badge info">{ext}</span>
                      ))}
                    </div>

                    <p style={{ marginTop: '1.5rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                      Supports Brain MRI (T1, T1c, T2, FLAIR)
                    </p>
                  </div>
                )}
              </div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    marginTop: '1.5rem', padding: '1rem 1.25rem',
                    background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)',
                    borderRadius: 'var(--radius-md)', display: 'flex',
                    alignItems: 'center', gap: '0.75rem',
                  }}
                >
                  <AlertCircle size={20} color="var(--color-danger)" />
                  <span style={{ fontSize: '0.875rem', color: 'var(--color-danger)' }}>{error}</span>
                </motion.div>
              )}
            </motion.div>
          ) : (
            /* Success Result */
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="glass-card-static" style={{ padding: '2rem', position: 'relative' }}>
                <button
                  onClick={reset}
                  style={{
                    position: 'absolute', top: '1rem', right: '1rem',
                    background: 'transparent', border: 'none', cursor: 'pointer',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  <X size={20} />
                </button>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <CheckCircle2 size={28} color="var(--color-accent)" />
                  <div>
                    <h3 style={{ fontWeight: 700, fontSize: '1.125rem' }}>Upload Successful</h3>
                    <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>{result.filename}</p>
                  </div>
                </div>

                <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
                  {[
                    { label: 'File ID', value: result.file_id, icon: FileImage },
                    { label: 'MRI Type', value: result.mri_type.toUpperCase(), icon: Brain },
                    { label: 'Modality', value: result.modality, icon: Eye },
                    { label: 'Shape', value: result.shape.join(' × ') },
                    { label: 'Size', value: `${result.file_size_mb} MB` },
                  ].map((item, i) => (
                    <div key={i} style={{
                      padding: '0.75rem', borderRadius: 'var(--radius-md)',
                      background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--color-border)',
                    }}>
                      <div style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: '0.9375rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--color-primary)' }}>
                        {item.value}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Preview */}
                {result.preview && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--color-text-secondary)' }}>
                      Preview — Multi-planar Views
                    </h4>
                    <img
                      src={result.preview}
                      alt="MRI Preview"
                      style={{
                        width: '100%', borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--color-border)',
                      }}
                    />
                  </div>
                )}

                <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <button className="btn btn-primary" onClick={reset}>
                    <UploadIcon size={16} /> Upload Another
                  </button>
                  <button className="btn btn-secondary" onClick={() => navigateWithFile('/analysis')}>
                    <BarChart3 size={16} /> Analyze
                  </button>
                  <button className="btn btn-secondary" onClick={() => navigateWithFile('/preprocessing')}>
                    <Sliders size={16} /> Preprocess
                  </button>
                  <button className="btn btn-secondary" onClick={() => navigateWithFile('/enhancement')}>
                    <Sparkles size={16} /> Enhance
                  </button>
                  <button className="btn btn-secondary" onClick={() => navigateWithFile('/segmentation')}>
                    <Layers size={16} /> Segment
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Supported Formats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="glass-card-static"
          style={{ padding: '1.5rem', marginTop: '2rem' }}
        >
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem' }}>Supported Input</h3>
          <div>
            <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--color-primary)', marginBottom: '0.5rem' }}>Brain MRI</h4>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {['T1', 'T1c', 'T2', 'FLAIR'].map(m => <span key={m} className="metric-badge info">{m}</span>)}
            </div>
          </div>
          <p style={{ marginTop: '1rem', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
            ⚠️ Full 3D NIfTI volumes only. Files are never converted to JPG/PNG — all processing uses native volumetric data.
          </p>
        </motion.div>
      </div>
    </div>
  )
}
