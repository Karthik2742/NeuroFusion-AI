import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Analysis from './pages/Analysis'
import Preprocessing from './pages/Preprocessing'
import Enhancement from './pages/Enhancement'
import Segmentation from './pages/Segmentation'
import Evaluation from './pages/Evaluation'
import Viewer from './pages/Viewer'
import Settings from './pages/Settings'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route element={<Layout />}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/preprocessing" element={<Preprocessing />} />
        <Route path="/enhancement" element={<Enhancement />} />
        <Route path="/segmentation" element={<Segmentation />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/viewer" element={<Viewer />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}

export default App
