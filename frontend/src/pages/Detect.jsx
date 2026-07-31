import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import UploadPanel from '../components/UploadPanel';
import PipelineVisualizer from '../components/PipelineVisualizer';
import ErrorMessage from '../components/ErrorMessage';
import Button from '../components/Button';
import LoadingSpinner from '../components/LoadingSpinner';
import { ToastContainer, useToast } from '../components/Toast';
import { uploadImage, preprocessImage, classifyImage } from '../services/api';

export default function Detect() {
  const navigate = useNavigate();
  const { toasts, addToast, dismissToast } = useToast();

  const [isLoading,        setIsLoading]        = useState(false);
  const [error,            setError]            = useState('');
  const [uploadedImage,    setUploadedImage]    = useState(null);
  const [preprocessResult, setPreprocessResult] = useState(null);
  const [activeStep,       setActiveStep]       = useState(null);
  const [completedSteps,   setCompletedSteps]   = useState(new Set());

  const markDone = (step) =>
    setCompletedSteps((prev) => new Set([...prev, step]));

  // ── Upload handler ────────────────────────────────────────────────────────
  const handleUpload = async (file, onProgress) => {
    setError('');
    setUploadedImage(null);
    setPreprocessResult(null);
    setCompletedSteps(new Set());
    setActiveStep('upload');
    setIsLoading(true);

    try {
      const res  = await uploadImage(file, (event) => {
        if (event.total) {
          const pct = Math.round((event.loaded / event.total) * 100);
          onProgress(pct);
        }
      });
      const data = res?.data ?? res;

      markDone('upload');
      setUploadedImage(data);
      addToast('success', `"${data.filename}" uploaded successfully.`);
    } catch (err) {
      setActiveStep(null);
      const msg = err.message || 'Upload failed. Please try again.';
      setError(msg);
      addToast('error', msg);
      throw err;
    } finally {
      setIsLoading(false);
      setActiveStep(null);
    }
  };

  // ── Auto-run full pipeline once upload completes ──────────────────────────
  useEffect(() => {
    if (!uploadedImage) return;
    runPipeline(uploadedImage.image_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadedImage]);

  const runPipeline = async (imageId) => {
    setError('');

    // ── Step 1: Preprocess (resize + ACEA + denoise) ─────────────────────────
    setActiveStep('resize');
    setIsLoading(true);
    try {
      const res  = await preprocessImage(imageId);
      const data = res?.data ?? res;
      // Mark resize → ACEA → denoise → segment → features all done
      // (backend runs them as one call; visualizer steps are informational)
      markDone('resize');
      markDone('preprocess');
      markDone('denoise');
      markDone('segment');
      markDone('features');
      setPreprocessResult(data);
      addToast('success', 'Preprocessing complete — ACEA · Median Filter · FCM · GLCM');
    } catch (err) {
      const msg = err.message || 'Preprocessing failed.';
      setError(msg);
      addToast('error', msg);
      setIsLoading(false);
      setActiveStep(null);
      return;
    }

    // ── Step 2: Classify (MambaVision / CNN via AI service) ──────────────────
    setActiveStep('classify');
    try {
      const res  = await classifyImage(imageId);
      const data = res?.data ?? res;
      markDone('classify');
      addToast('success', `AI prediction: ${data.predicted_class} (${(data.confidence * 100).toFixed(1)}%)`);
      navigate(`/results?imageId=${imageId}`);
    } catch (err) {
      const msg = err.message || 'Classification failed.';
      setError(msg);
      addToast('error', msg);
    } finally {
      setIsLoading(false);
      setActiveStep(null);
    }
  };

  // ── Reset everything ──────────────────────────────────────────────────────
  const handleReset = () => {
    setUploadedImage(null);
    setPreprocessResult(null);
    setError('');
    setActiveStep(null);
    setCompletedSteps(new Set());
  };

  const showUploadPanel  = !uploadedImage;
  const showSpinner      = isLoading;
  const showPreprocessed = !!preprocessResult && !isLoading && activeStep !== 'classify';

  return (
    <Layout>
      <div className="max-w-3xl mx-auto space-y-6">

        {/* Page header */}
        <div>
          <h1 className="text-2xl font-bold text-pipeline-900">Tumor Detection</h1>
          <p className="text-pipeline-500 mt-1 text-sm">
            Upload an MRI scan to run the full detection pipeline.
          </p>
        </div>

        {/* Pipeline step tracker */}
        <div className="card">
          <h2 className="section-title">Pipeline Progress</h2>
          <PipelineVisualizer activeStep={activeStep} completedSteps={completedSteps} />
        </div>

        {/* Upload panel */}
        {showUploadPanel && (
          <div className="card">
            <h2 className="section-title">Upload MRI Image</h2>
            <UploadPanel onUpload={handleUpload} isLoading={isLoading} />
          </div>
        )}

        {/* Error message */}
        {error && (
          <ErrorMessage message={error} onRetry={handleReset} />
        )}

        {/* Processing spinner */}
        {showSpinner && uploadedImage && (
          <div className="card flex items-center gap-4 py-6">
            <LoadingSpinner size="md" message="" />
            <div>
              <p className="font-semibold text-pipeline-800">
                {activeStep === 'resize'   && 'Resizing to 256×256 · ACEA · Median Filter…'}
                {activeStep === 'classify' && 'Running MambaVision inference…'}
                {!activeStep               && 'Processing…'}
              </p>
              <p className="text-xs text-pipeline-400 mt-0.5">
                {activeStep === 'resize'   && 'Adaptive contrast stretch · denoising · FCM segmentation · GLCM features'}
                {activeStep === 'classify' && 'Deep learning classification + Grad-CAM heatmap generation'}
              </p>
            </div>
          </div>
        )}

        {/* Preprocessing preview (shown briefly before classify starts) */}
        {showPreprocessed && (
          <div className="card space-y-5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24"
                     stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round"
                        d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                </svg>
              </div>
              <div>
                <p className="font-semibold text-pipeline-900">Preprocessing complete — running AI inference…</p>
                <p className="text-xs text-pipeline-400 mt-0.5">
                  Resize (256×256) · ACEA · {preprocessResult.computational_time_ms?.toFixed(0)} ms
                </p>
              </div>
            </div>
          </div>
        )}

      </div>

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </Layout>
  );
}
