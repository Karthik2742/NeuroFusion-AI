import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Sparkles, BarChart3, Box, Shield, Zap, ArrowRight, Cpu, Sun, Moon } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

const features = [
  {
    icon: Brain,
    title: 'MRI Analysis',
    description: 'Auto-detect Brain MRI types, modalities, and generate comprehensive dataset statistics.',
    color: '#00d4ff',
  },
  {
    icon: Sparkles,
    title: 'AI Enhancement',
    description: 'Restormer, SwinIR, MIRNet — state-of-the-art deep learning models for MRI quality enhancement.',
    color: '#7c3aed',
  },
  {
    icon: BarChart3,
    title: 'Evaluation Metrics',
    description: 'PSNR, SSIM, FSIM, BRISQUE, NIQE and 15+ metrics for rigorous quality assessment.',
    color: '#10b981',
  },
  {
    icon: Box,
    title: '3D Visualization',
    description: 'Interactive NIfTI volume rendering with multi-planar views and segmentation overlays.',
    color: '#f59e0b',
  },
  {
    icon: Shield,
    title: 'Disease Localization',
    description: 'Automatic detection of tumors, edema, and lesions with heatmap overlays.',
    color: '#ef4444',
  },
  {
    icon: Zap,
    title: 'GPU Accelerated',
    description: 'CUDA-optimized pipelines with mixed precision, gradient checkpointing, and real-time inference.',
    color: '#ec4899',
  },
]

const stats = [
  { value: '4+', label: 'Enhancement Models' },
  { value: '7+', label: 'Segmentation Architectures' },
  { value: '15+', label: 'Quality Metrics' },
  { value: '3D', label: 'Volume Rendering' },
]

export default function Landing() {
  const navigate = useNavigate()
  const { theme, toggleTheme } = useTheme()

  return (
    <div style={{ minHeight: '100vh', background: 'var(--gradient-dark)', overflow: 'hidden' }}>
      {/* Background Effects */}
      <div style={{
        position: 'fixed', inset: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse at 30% 20%, rgba(0, 212, 255, 0.06) 0%, transparent 50%), radial-gradient(ellipse at 70% 80%, rgba(124, 58, 237, 0.06) 0%, transparent 50%)',
      }} />
      <div className="bg-grid" style={{ position: 'fixed', inset: 0, pointerEvents: 'none', opacity: 0.4 }} />

      {/* Navigation */}
      <motion.nav
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        style={{
          position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', padding: '1.25rem 3rem',
          borderBottom: '1px solid var(--color-border)',
          background: 'var(--header-bg)', backdropFilter: 'blur(10px)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: 40, height: 40, background: 'var(--gradient-primary)',
            borderRadius: 'var(--radius-md)', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
          }}>
            <Brain size={22} color="white" />
          </div>
          <div>
            <span style={{ fontWeight: 800, fontSize: '1.125rem', background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              NeuroFusion AI
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            className="btn btn-ghost"
            onClick={toggleTheme}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem' }}
          >
            {theme === 'dark' ? <Sun size={16} color="#f59e0b" /> : <Moon size={16} color="#7c3aed" />}
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
          <button className="btn btn-ghost" onClick={() => navigate('/dashboard')}>Dashboard</button>
          <button className="btn btn-primary" onClick={() => navigate('/upload')}>
            Get Started <ArrowRight size={16} />
          </button>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section style={{ position: 'relative', zIndex: 5, maxWidth: 1200, margin: '0 auto', padding: '6rem 2rem 4rem', textAlign: 'center' }}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
        >
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
            padding: '0.375rem 1rem', borderRadius: 'var(--radius-full)',
            background: 'var(--color-primary-glow)', border: '1px solid rgba(0, 212, 255, 0.15)',
            marginBottom: '2rem', fontSize: '0.8125rem', color: 'var(--color-primary)',
          }}>
            <Cpu size={14} />
            Research-Grade AI Medical Imaging Platform
          </div>

          <h1 style={{
            fontSize: 'clamp(2.5rem, 6vw, 4.5rem)', fontWeight: 900, lineHeight: 1.1,
            marginBottom: '1.5rem', letterSpacing: '-0.02em',
          }}>
            <span style={{
              background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent', backgroundSize: '200% auto',
              animation: 'gradient-shift 3s ease infinite',
            }}>
              BRAIN TUMOR DETECTION
            </span>
            <br />
            <span style={{ color: 'var(--color-text-primary)' }}>using </span>
            <span style={{
              background: 'var(--gradient-secondary)', WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              MONAI
            </span>
          </h1>

          <p style={{
            fontSize: '1.125rem', color: 'var(--color-text-secondary)', maxWidth: 700,
            margin: '0 auto 2.5rem', lineHeight: 1.7,
          }}>
            Analyze, enhance, and segment Brain MRI volumes using Restormer, SwinIR, MIRNet,
            and 7+ segmentation architectures. Built with PyTorch, MONAI, and FastAPI.
          </p>

          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn btn-primary btn-lg"
              onClick={() => navigate('/upload')}
              style={{ fontSize: '1rem', padding: '1rem 2.5rem' }}
            >
              Upload MRI <ArrowRight size={18} />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn btn-secondary btn-lg"
              onClick={() => navigate('/dashboard')}
            >
              View Dashboard
            </motion.button>
          </div>
        </motion.div>

        {/* Stats Row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem',
            maxWidth: 800, margin: '4rem auto 0',
          }}
        >
          {stats.map((stat, i) => (
            <div key={i} style={{ textAlign: 'center' }}>
              <div style={{
                fontSize: '2.5rem', fontWeight: 900, fontFamily: 'var(--font-mono)',
                background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                {stat.value}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Features Grid */}
      <section style={{ position: 'relative', zIndex: 5, maxWidth: 1200, margin: '0 auto', padding: '2rem 2rem 6rem' }}>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          <h2 style={{ textAlign: 'center', fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>
            <span style={{ background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Platform Capabilities
            </span>
          </h2>
          <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', marginBottom: '3rem', fontSize: '1rem' }}>
            A complete end-to-end pipeline for medical MRI research
          </p>

          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
            gap: '1.5rem',
          }}>
            {features.map((feature, i) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.8 + i * 0.1 }}
                  className="glass-card"
                  style={{ padding: '2rem', cursor: 'default' }}
                >
                  <div style={{
                    width: 48, height: 48, borderRadius: 'var(--radius-md)',
                    background: `${feature.color}15`, border: `1px solid ${feature.color}30`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    marginBottom: '1rem',
                  }}>
                    <Icon size={24} color={feature.color} />
                  </div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '0.5rem', color: 'var(--color-text-primary)' }}>
                    {feature.title}
                  </h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                    {feature.description}
                  </p>
                </motion.div>
              )
            })}
          </div>
        </motion.div>
      </section>

      {/* Tech Stack Bar */}
      <section style={{
        borderTop: '1px solid var(--color-border)', padding: '2rem',
        textAlign: 'center', position: 'relative', zIndex: 5,
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: '2rem', flexWrap: 'wrap', opacity: 0.5,
          fontSize: '0.8125rem', color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)',
        }}>
          {['PyTorch', 'MONAI', 'FastAPI', 'React', 'TypeScript', 'CUDA', 'NIfTI', 'SimpleITK'].map((tech) => (
            <span key={tech}>{tech}</span>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--color-border)', padding: '2rem',
        textAlign: 'center', position: 'relative', zIndex: 5,
      }}>
        <p style={{ fontSize: '0.8125rem', color: 'var(--color-text-muted)' }}>
          NeuroFusion AI — Research & Demonstration Platform. Not for clinical diagnosis.
        </p>
      </footer>
    </div>
  )
}
