import { useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import Button from '../components/Button';
import Badge from '../components/Badge';

const PIPELINE_STEPS = [
  { label: 'MRI Upload',      desc: 'T1-mode image input' },
  { label: 'Resize',          desc: '256 × 256 px' },
  { label: 'ACEA',            desc: 'Contrast enhancement' },
  { label: 'Median Filter',   desc: 'Noise removal' },
  { label: 'FCM',             desc: 'Segmentation (C=3)' },
  { label: 'GLCM',            desc: '7 texture features' },
  { label: 'MambaVision',     desc: 'Classification' },
];

const METRICS = [
  { label: 'Accuracy',    value: '98.51%' },
  { label: 'Sensitivity', value: '98.2%' },
  { label: 'Specificity', value: '98.9%' },
  { label: 'PSNR',        value: '52.98 dB' },
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <Layout>
      <div className="space-y-8 max-w-4xl mx-auto">

        {/* Hero */}
        <section className="card text-center py-12">
          <Badge variant="info" label="Research Implementation" />
          <h1 className="mt-4 text-3xl sm:text-4xl font-bold text-pipeline-900 leading-tight">
            MRI Brain Tumor Detection
          </h1>
          <p className="mt-3 text-pipeline-500 text-lg max-w-2xl mx-auto">
            Using MambaVision Transformer with CNN backbone for highly accurate
            classification of brain MRI scans.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row gap-3 justify-center">
            <Button variant="primary" onClick={() => navigate('/detect')}>
              Start Detection
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
              </svg>
            </Button>
            <Button variant="secondary" onClick={() => navigate('/results')}>
              View Results
            </Button>
          </div>
        </section>

        {/* Paper citation */}
        <section className="card border-l-4 border-l-blue-500">
          <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-1">Paper Reference</p>
          <p className="text-sm text-pipeline-700 leading-relaxed">
            Anantharajan et al., <em>"Brain Tumour Detection using Enhanced Deep Neural Network with Support Vector Machine"</em>,{' '}
            <span className="font-medium">Measurement: Sensors 31 (2024)</span> — extended with MambaVision Transformer backbone.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge variant="info" label="255 MRI Images" />
            <Badge variant="info" label="T1-mode" />
            <Badge variant="info" label="Kaggle Dataset" />
            <Badge variant="info" label="98 Healthy · 155 Tumor" />
          </div>
        </section>

        {/* Pipeline overview */}
        <section className="card">
          <h2 className="section-title">Detection Pipeline</h2>
          <div className="flex flex-wrap items-center gap-2">
            {PIPELINE_STEPS.map((step, idx) => (
              <div key={step.label} className="flex items-center gap-2">
                <div className="flex flex-col items-center">
                  <div className="bg-blue-600 text-white text-xs font-bold w-8 h-8 rounded-full flex items-center justify-center">
                    {idx + 1}
                  </div>
                  <p className="text-xs font-semibold text-pipeline-800 mt-1 text-center w-20 leading-tight">{step.label}</p>
                  <p className="text-xs text-pipeline-400 text-center w-20 leading-tight hidden sm:block">{step.desc}</p>
                </div>
                {idx < PIPELINE_STEPS.length - 1 && (
                  <svg className="w-4 h-4 text-pipeline-300 flex-shrink-0 mb-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                  </svg>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Performance targets */}
        <section className="card">
          <h2 className="section-title">Performance</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {METRICS.map(({ label, value }) => (
              <div key={label} className="bg-pipeline-50 rounded-lg p-4 text-center border border-pipeline-100">
                <p className="text-2xl font-bold text-blue-600">{value}</p>
                <p className="text-xs text-pipeline-500 mt-1 font-medium">{label}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Dataset info */}
        <section className="card">
          <h2 className="section-title">Dataset</h2>
          <div className="grid sm:grid-cols-3 gap-4 text-sm">
            <div className="space-y-1">
              <p className="font-semibold text-pipeline-700">Source</p>
              <p className="text-pipeline-500">Kaggle Brain MRI Dataset</p>
            </div>
            <div className="space-y-1">
              <p className="font-semibold text-pipeline-700">Modality</p>
              <p className="text-pipeline-500">T1-weighted MRI scans</p>
            </div>
            <div className="space-y-1">
              <p className="font-semibold text-pipeline-700">Distribution</p>
              <p className="text-pipeline-500">255 total · 98 healthy · 155 tumor</p>
            </div>
          </div>
        </section>

      </div>
    </Layout>
  );
}
