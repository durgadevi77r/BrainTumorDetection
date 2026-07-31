/**
 * SinglePrediction — upload one MRI scan and view the full inference result.
 */

import { useState } from 'react';
import Layout from '@/components/Layout';
import UploadPanel from '@/components/UploadPanel';
import PredictionCard from '@/components/PredictionCard';
import ProbabilityChart from '@/components/ProbabilityChart';
import GradCAMViewer from '@/components/GradCAMViewer';
import ModelSelector from '@/components/ModelSelector';
import LoadingSpinner from '@/components/LoadingSpinner';
import { ToastContainer, useToast } from '@/components/Toast';
import { usePrediction } from '@/hooks/usePrediction';
import { useModels } from '@/hooks/useModels';
import type { ArchitectureName } from '@/types';

export default function SinglePrediction() {
  const { predict, result, loading, error, uploadProgress, reset } = usePrediction();
  const { models } = useModels();
  const { toasts, addToast, dismissToast } = useToast();

  const [modelName, setModelName] = useState<ArchitectureName>('mambavision');
  const [topK, setTopK] = useState(3);
  const [generateGradcam, setGenerateGradcam] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    // Build local preview
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    try {
      await predict(file, { modelName, topK, generateGradcam, confidenceThreshold: 0.5 });
      addToast('success', 'Prediction complete');
    } catch {
      addToast('error', error?.detail ?? 'Prediction failed');
    }
  };

  const handleReset = () => {
    reset();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  };

  return (
    <Layout>
      <div className="max-w-3xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-pipeline-900">Single Prediction</h1>
          <p className="text-pipeline-500 mt-1 text-sm">Upload one MRI scan to classify it instantly.</p>
        </div>

        {/* Config */}
        {!result && (
          <div className="card">
            <h2 className="section-title">Inference Settings</h2>
            <div className="grid sm:grid-cols-3 gap-4">
              <ModelSelector value={modelName} onChange={setModelName} models={models} />
              <div className="space-y-1.5">
                <label htmlFor="top-k" className="block text-xs font-semibold text-pipeline-600 uppercase tracking-wide">Top-K</label>
                <select id="top-k" value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full rounded-lg border border-pipeline-200 bg-white px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {[1,2,3,4].map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </div>
              <div className="flex items-end pb-0.5">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input type="checkbox" checked={generateGradcam} onChange={(e) => setGenerateGradcam(e.target.checked)}
                    className="w-4 h-4 rounded text-blue-600 border-pipeline-300 focus:ring-blue-500" />
                  <span className="text-sm text-pipeline-700">Generate Grad-CAM</span>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Upload */}
        {!result && !loading && (
          <div className="card">
            <h2 className="section-title">Upload MRI Image</h2>
            {/* Hidden direct input for programmatic/test access */}
            <input
              type="file"
              accept="image/jpeg,image/png"
              aria-label="Upload MRI image"
              className="sr-only"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }}
            />
            <UploadPanel onUpload={handleUpload} isLoading={loading} autoSubmit />
          </div>
        )}

        {/* Progress */}
        {loading && (
          <div className="card flex items-center gap-4 py-6">
            <LoadingSpinner size="md" message="" />
            <div className="flex-1">
              <p className="font-semibold text-pipeline-800">Running inference…</p>
              {uploadProgress < 100 && (
                <div className="mt-2">
                  <div className="h-1.5 w-full bg-pipeline-200 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${uploadProgress}%` }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="card bg-red-50 border-red-200">
            <p className="text-sm font-semibold text-red-700">Prediction Error</p>
            <p className="text-sm text-red-600 mt-1">{error.detail}</p>
            <button onClick={handleReset} className="mt-3 text-xs font-medium text-red-700 underline">Try again</button>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="space-y-4">
            <PredictionCard result={result} showTopK showTiming showGradcam />
            <div className="card"><ProbabilityChart result={result} /></div>
            {generateGradcam && (
              <div className="card">
                <GradCAMViewer gradcamPath={result.metadata.gradcam_path} originalSrc={previewUrl ?? undefined} />
              </div>
            )}
            <button onClick={handleReset}
              className="w-full text-center text-sm font-medium text-pipeline-500 hover:text-pipeline-800 py-2 border border-pipeline-200 rounded-xl hover:bg-pipeline-50 transition-colors">
              ← Predict Another Image
            </button>
          </div>
        )}
      </div>
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </Layout>
  );
}
