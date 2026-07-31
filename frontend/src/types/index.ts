/**
 * src/types/index.ts — Shared TypeScript models mirroring the FastAPI schemas.
 *
 * These types are derived from:
 *   app/inference/results.py
 *   training/experiment.py
 *   app/dataset/metadata.py
 *   app/preprocessing/quality.py
 */

// ─── Inference / Prediction ────────────────────────────────────────────────

export interface PredictionMetadata {
  model_name: string;
  model_version: string | null;
  image_size: number;
  class_names: string[];
  predicted_at: string;         // ISO-8601
  source_path: string | null;
  gradcam_path: string | null;
}

export interface TopKPrediction {
  rank: number;
  class_name: string;
  class_index: number;
  probability: number;          // 0–1, 4 d.p.
}

export interface PredictionResult {
  image_id: string;
  predicted_class: string;
  predicted_class_index: number;
  confidence: number;           // 0–1
  is_high_confidence: boolean;
  probabilities: Record<string, number>;
  top_k: TopKPrediction[];
  timing_ms: number;
  metadata: PredictionMetadata;
  error: string | null;
}

export interface BatchItemResult {
  filename: string;
  success: boolean;
  result: PredictionResult | null;
  error: string | null;
}

export interface BatchPredictionResult {
  total: number;
  succeeded: number;
  failed: number;
  success_rate: number;
  timing_ms: number;
  model_name: string;
  source_type: 'directory' | 'zip' | 'list';
  class_distribution: Record<string, number>;
  export_paths: Record<string, string>;
  results: BatchItemResult[];
}

// ─── Training ─────────────────────────────────────────────────────────────

export type TrainingStatus = 'queued' | 'running' | 'completed' | 'failed' | 'interrupted' | 'created';

export interface TrainingJob {
  job_id: string;
  status: TrainingStatus;
  experiment_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  config?: Record<string, unknown>;
}

export interface TrainingStartRequest {
  architecture: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  dataset_dir?: string | null;
  fine_tune: boolean;
  fine_tune_layers: number;
  fine_tune_epochs: number;
  fine_tune_lr?: number | null;
  class_weights?: Record<string, number> | null;
  seed: number;
}

export interface EpochHistory {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

export interface Experiment {
  experiment_id: string;
  architecture: string;
  status: TrainingStatus;
  created_at: string;
  finished_at: string | null;
  duration_s: number | null;
  epochs_trained: number | null;
  best_val_accuracy: number | null;
  notes: string;
  config?: Record<string, unknown>;
  history?: EpochHistory[];
  evaluation?: Record<string, unknown>;
}

// ─── Models ───────────────────────────────────────────────────────────────

export type ArchitectureName = 'mambavision' | 'cnn' | 'vgg16' | 'resnet50' | 'efficientnet';

export interface ModelInfo {
  name: ArchitectureName;
  available: boolean;
  cached: boolean;
  model_version: string | null;
  total_params: number | null;
  model_dir: string;
}

export interface CacheStats {
  capacity: number;
  size: number;
  cached_models: string[];
  total_hits: number;
  total_misses: number;
  hit_rate: number;
  entries: Array<{
    model_name: string;
    loaded_at: string;
    last_accessed_at: string;
    hit_count: number;
    load_duration_ms: number;
    total_params: number;
    model_version: string | null;
  }>;
}

export interface ActiveModelInfo {
  model_name: string;
  available: boolean;
  cached: boolean;
  model_info: Record<string, unknown>;
  cache_stats: CacheStats;
}

// ─── Dataset ──────────────────────────────────────────────────────────────

export interface ClassSplit {
  train: number;
  val: number;
  test: number;
}

export interface DatasetInfo {
  classes: string[];
  class_to_index: Record<string, number>;
  total_per_split: { train: number; val: number; test: number };
  total_images: number;
  imbalance_ratio: number;
  is_balanced: boolean;
  per_class_counts?: Record<string, ClassSplit>;
  pixel_mean?: number[];
  pixel_std?: number[];
  processed_dir?: string;
  created_at?: string;
}

export interface DatasetValidationReport {
  is_valid: boolean;
  classes_found: string[];
  class_counts: Record<string, number>;
  total_images: number;
  errors: string[];
  warnings: string[];
}

export interface DatasetPrepareRequest {
  raw_dir?: string | null;
  output_dir?: string | null;
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
  seed: number;
  overwrite: boolean;
  full_stats: boolean;
}

// ─── Preprocessing ────────────────────────────────────────────────────────

export interface QualityCheck {
  name: string;
  passed: boolean;
  value: number;
  threshold: number;
  message: string;
}

export interface QualityReport {
  is_valid: boolean;
  image_width: number;
  image_height: number;
  file_size_bytes: number;
  checks: QualityCheck[];
  warnings: string[];
  errors: string[];
}

export interface PreprocessPreviewResult {
  quality: QualityReport;
  config: Record<string, unknown>;
  preprocessed_b64: string;
  augmented_b64: string[];
}

// ─── API response envelope ────────────────────────────────────────────────

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface ApiError {
  detail: string;
  status: number;
}

// ─── Dashboard / Metrics ──────────────────────────────────────────────────

export interface DashboardAlert {
  level: 'warning' | 'critical';
  domain: string;
  message: string;
}

export interface SystemMetrics {
  timestamp: string;
  uptime_seconds: number;
  platform: string;
  python_version: string;
  // CPU
  cpu_percent: number | null;
  cpu_per_core: number[];
  cpu_count_logical: number | null;
  cpu_count_physical: number | null;
  cpu_freq_mhz: number | null;
  // RAM
  ram_total_mb: number | null;
  ram_used_mb: number | null;
  ram_available_mb: number | null;
  ram_percent: number | null;
  // Disk
  disk_total_gb: number | null;
  disk_used_gb: number | null;
  disk_free_gb: number | null;
  disk_percent: number | null;
  // GPU
  gpu_available: boolean;
  gpu_count: number;
  gpus: Array<{
    index: number;
    name: string;
    memory_used_mb: number | null;
    memory_total_mb: number | null;
    utilization_percent: number | null;
  }>;
  // Process
  process_pid: number | null;
  process_cpu_percent: number | null;
  process_ram_mb: number | null;
  process_threads: number | null;
}

export interface ConfidenceDistribution {
  buckets: string[];
  counts: number[];
}

export interface TopClass {
  class_name: string;
  count: number;
}

export interface RecentPrediction {
  image_id: string | null;
  model_name: string;
  predicted_class: string | null;
  confidence: number | null;
  timing_ms: number;
  success: boolean;
  timestamp: string;
}

export interface InferenceMetrics {
  timestamp: string;
  started_at: string;
  total_predictions: number;
  succeeded: number;
  failed: number;
  success_rate: number;
  per_model_counts: Record<string, number>;
  avg_latency_ms: number | null;
  min_latency_ms: number | null;
  max_latency_ms: number | null;
  p95_latency_ms: number | null;
  confidence_distribution: ConfidenceDistribution;
  class_distribution: Record<string, number>;
  top_classes: TopClass[];
  batch_runs: number;
  batch_images_processed: number;
  batch_succeeded: number;
  batch_failed: number;
  recent_predictions: RecentPrediction[];
}

export interface RecentJob {
  job_id: string;
  status: string;
  architecture: string;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
}

export interface RecentExperiment {
  experiment_id: string | null;
  architecture: string | null;
  status: string | null;
  best_val_accuracy: number | null;
  epochs_trained: number | null;
  duration_s: number | null;
  created_at: string | null;
}

export interface TrainingMetrics {
  timestamp: string;
  total_jobs: number;
  queued_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  avg_job_duration_s: number | null;
  architecture_counts: Record<string, number>;
  recent_jobs: RecentJob[];
  total_experiments: number;
  best_val_accuracy: number | null;
  recent_experiments: RecentExperiment[];
}

export interface DashboardOverviewSystem {
  cpu_percent: number | null;
  ram_percent: number | null;
  ram_used_mb: number | null;
  disk_percent: number | null;
  gpu_available: boolean;
  uptime_seconds: number | null;
  platform: string | null;
}

export interface DashboardOverviewInference {
  total_predictions: number;
  succeeded: number;
  failed: number;
  success_rate: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  top_classes: TopClass[];
  batch_runs: number;
}

export interface DashboardOverviewTraining {
  total_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  best_val_accuracy: number | null;
  total_experiments: number;
}

export interface DashboardOverview {
  timestamp: string;
  service_version: string;
  system: DashboardOverviewSystem;
  inference: DashboardOverviewInference;
  training: DashboardOverviewTraining;
  models: Record<string, unknown>;
  alerts: DashboardAlert[];
}

export interface DashboardHistoryPoint {
  timestamp: string;
  [key: string]: unknown;
}

export interface DashboardHistory {
  metric_type: string;
  hours: number;
  count: number;
  data: DashboardHistoryPoint[];
}

// ─── Health ───────────────────────────────────────────────────────────────

export interface HealthResponse {
  success: boolean;
  status: string;
  service: string;
  version: string;
  timestamp: string;
  environment: string;
  active_model: string;
  class_names: string[];
  image_size: number;
  python_version: string;
  models_available: Record<string, boolean>;
}
