import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL !== undefined 
  ? import.meta.env.VITE_API_URL 
  : (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? '' : 'http://localhost:8000');

export const api = axios.create({
  baseURL: API_BASE ? `${API_BASE}/api` : '/api',
  timeout: 300000, // 5 min for large MRI files
  headers: {
    'Accept': 'application/json',
  },
});

// Types
export interface UploadResponse {
  file_id: string;
  filename: string;
  file_size_mb: number;
  shape: number[];
  modality: string;
  mri_type: string;
  preview?: string;
}

export interface AnalysisResponse {
  file_id: string;
  mri_type: string;
  modality: string;
  intensity_stats: Record<string, any>;
  spatial_stats: Record<string, any>;
  quality_metrics: Record<string, number>;
  charts: Record<string, string>;
}

export interface EnhanceResponse {
  file_id: string;
  model_used: string;
  output_file_id?: string;
  metrics: Record<string, number>;
  inference_time_sec: number;
  peak_memory_gb: number;
  device: string;
  comparison_view?: string;
  preview?: string;
}

export interface LocalizeResponse {
  file_id: string;
  mri_type: string;
  localization: {
    detected: boolean;
    abnormal_volume_ratio: number;
    findings: Array<{
      pathology: string;
      confidence: number;
      description: string;
      recommendation: string;
    }>;
    disclaimer: string;
  };
  heatmap_preview?: string;
}

export interface RegionMeasurement {
  name: string;
  description: string;
  color: string;
  voxels: number;
  volume_cm3: number;
  volume_mm3: number;
}

export interface SegmentResponse {
  file_id: string;
  architecture: string;
  mask_file_id: string;
  overlay_preview?: string;
  shape?: number[];
  voxel_spacing?: number[];
  best_slice_idx?: number;
  total_slices?: number;
  measurements?: {
    wt: RegionMeasurement;
    tc: RegionMeasurement;
    et: RegionMeasurement;
  };
  validation?: {
    valid_nesting: boolean;
    warning?: string | null;
    message: string;
  };
  tumor_status?: string;
  assessment?: {
    tumor_status: string;
    explanation: string;
    validation_status: string;
    technical_warnings: string[];
  };
  classes_detected: Record<string, number>;
}

export interface GPUStatus {
  is_available: boolean;
  device_name: string;
  total_memory_gb: number;
  allocated_memory_gb: number;
  free_memory_gb: number;
  utilization_percent: number;
  cuda_version: string;
}

export interface FileInfo {
  file_id: string;
  filename: string;
  shape: number[];
  modality: string;
  mri_type: string;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  parameters: { total: number; trainable: number };
}

// API Functions
export const uploadMRI = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const analyzeMRI = async (fileId: string): Promise<AnalysisResponse> => {
  const { data } = await api.post('/analyze', { file_id: fileId });
  return data;
};

export const preprocessMRI = async (fileId: string, steps: string[]): Promise<any> => {
  const { data } = await api.post('/preprocess', { file_id: fileId, steps });
  return data;
};

export const enhanceMRI = async (
  fileId: string,
  model: string = 'clahe',
  batchSize: number = 4,
): Promise<EnhanceResponse> => {
  const { data } = await api.post('/enhance', {
    file_id: fileId,
    model,
    batch_size: batchSize,
    compute_metrics: true,
  });
  return data;
};

export const localizeMRI = async (fileId: string): Promise<LocalizeResponse> => {
  const { data } = await api.post(`/localize?file_id=${fileId}`);
  return data;
};

export const segmentMRI = async (fileId: string, architecture: string = 'monai_unet3d'): Promise<SegmentResponse> => {
  const { data } = await api.post(`/segment?file_id=${fileId}&architecture=${architecture}`);
  return data;
};

export const getSegmentSliceUrl = (
  fileId: string,
  maskFileId: string,
  sliceIdx: number,
  plane: string = 'axial',
  showWt: boolean = true,
  showTc: boolean = true,
  showEt: boolean = true,
  alpha: number = 0.45
): string => {
  const base = API_BASE ? `${API_BASE}/api` : '/api';
  return `${base}/segment/slice?file_id=${fileId}&mask_file_id=${maskFileId}&slice_idx=${sliceIdx}&plane=${plane}&show_wt=${showWt}&show_tc=${showTc}&show_et=${showEt}&alpha=${alpha}`;
};

export const evaluateMRI = async (originalId: string, enhancedId: string): Promise<any> => {
  const { data } = await api.post('/evaluate', {
    original_id: originalId,
    enhanced_id: enhancedId,
  });
  return data;
};

export const getReportUrl = (fileId: string, maskFileId?: string, print: boolean = false): string => {
  const params = new URLSearchParams({ file_id: fileId });
  if (maskFileId) params.append('mask_file_id', maskFileId);
  if (print) params.append('print', 'true');
  return `${API_BASE}/api/report?${params.toString()}`;
};

export const getReportPdfUrl = (fileId: string, maskFileId?: string): string => {
  const params = new URLSearchParams({ file_id: fileId });
  if (maskFileId) params.append('mask_file_id', maskFileId);
  return `${API_BASE}/api/report/pdf?${params.toString()}`;
};

export const getGPUStatus = async (): Promise<GPUStatus> => {
  const { data } = await api.get('/gpu');
  return data;
};

export const getFiles = async (): Promise<{ files: FileInfo[] }> => {
  const { data } = await api.get('/files');
  return data;
};

export const getEnhancementModels = async (): Promise<{ models: ModelInfo[] }> => {
  const { data } = await api.get('/models/enhancement');
  return data;
};

export const downloadFile = (fileId: string): string => {
  return `${API_BASE}/api/download/${fileId}`;
};
