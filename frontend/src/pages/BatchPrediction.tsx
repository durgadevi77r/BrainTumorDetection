/**
 * BatchPrediction — multi-file or ZIP archive inference page.
 */

import { useState } from 'react';
import Layout from '@/components/Layout';
import BatchUpload, { type BatchMode } from '@/components/BatchUpload';
import PredictionTable from '@/components/PredictionTable';
import ModelSelector from '@/components/ModelSelector';
import LoadingSpinner from '@/components/LoadingSpinner';
import { ToastContainer, useToast } from '@/components/Toast';
import { useBatchPrediction } from '@/hooks/useBatchPrediction';
import { useModels } from '@/hooks/useModels';
import type { ArchitectureName } from '@/types';

export default function BatchPrediction() {
  const { runBatch, runZip, result, loading, error, reset } = useBatchPrediction();
  const { models } = useModels();
  const { toasts, addToast, dismissToast } = useToast();

  const [mode, setMode] = useState<BatchMode>('files');
  const [modelName, setModelName] = useState<ArchitectureName>('mambavision');
  const [topK, setTopK] = useState(1);

  const handleSubmit = async (payload: File | File[]) => {
    try {
      if (mode === 'zip') {
        await runZip(payload as File, { modelName, topK });
      } else {
        await runBatch(payload as File[], { modelName, topK });
      }
      addToast('success', 'Batch inference complete');
    } catch {
      addToast('error', error?.detail ?? 'Batch inference failed');
    }
  };

  const downloadJSON = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'batch_results.json'; a.click();
    URL.revokeObjectURL(url);
  };

  const downloadCSV = () => {
    if (!result) return;
    const header = 'filename,success,predicted_class,confidence,timing_ms,error\n';
    const rows = result.results.map((r) =>
      [r.filename, r.success, r.result?.predicted_class ?? '', r.result?.confidence ?? '', r.result?.timing_ms ?? '', r.error ?? ''].join(',')
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'batch_results.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Layout>
      <div className="max-w-5xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-pipeline-900">Batch Prediction</h1>
          <p className="text-pipeline-500 mt-1 text-sm">Upload multiple MRI images or a ZIP archive for bulk inference.</p>
        </div>

        {/* Config */}
        {!result && (
          <div className="card space-y-4">
            <h2 className="section-title">Settings</h2>

            {/* Mode toggle */}
            <div className="flex gap-2">
              {(['files', 'zip'] as BatchMode[]).map((m) => (
                <button key={m} onClick={() => { setMode(m); reset(); }}
                  className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors
                    ${mode === m ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-pipeline-600 border-pipeline-200 hover:bg-pipeline-50'}`}>
                  {m === 'files' ? '🖼 Multiple Files' : '🗜 ZIP Archive'}
                </button>
              ))}
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              <ModelSelector value={modelName} onChange={setModelName} models={models} />
              <div className="space-y-1.5">
                <label htmlFor="batch-topk" className="block text-xs font-semibold text-pipeline-600 uppercase tracking-wide">Top-K</label>
                <select id="batch-topk" value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full rounded-lg border border-pipeline-200 bg-white px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {[1,2,3,4].map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Upload */}
        {!result && (
          <div className="card">
            <BatchUpload mode={mode} onSubmit={handleSubmit} loading={loading} />
          </div>
        )}

        {/* Spinner */}
        {loading && (
          <div className="card flex items-center gap-4 py-6">
            <LoadingSpinner size="md" message="" />
            <p className="font-semibold text-pipeline-800">Running batch inference…</p>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="card bg-red-50 border-red-200">
            <p className="text-sm font-semibold text-red-700">{error.detail}</p>
            <button onClick={reset} className="mt-2 text-xs underline text-red-600">Try again</button>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="card space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="section-title mb-0">Results</h2>
              <button onClick={reset} className="text-sm text-pipeline-500 hover:text-pipeline-800 underline">
                ← New Batch
              </button>
            </div>

            {/* Class distribution */}
            <div className="flex flex-wrap gap-2">
              {Object.entries(result.class_distribution).map(([cls, count]) => (
                <span key={cls} className="text-xs font-medium bg-pipeline-100 text-pipeline-700 px-2.5 py-1 rounded-full border border-pipeline-200">
                  {cls}: {count}
                </span>
              ))}
            </div>

            <PredictionTable result={result} onDownloadCSV={downloadCSV} onDownloadJSON={downloadJSON} />
          </div>
        )}
      </div>
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </Layout>
  );
}
