/**
 * ResultCard
 *
 * Displays the AI classification result for a single MRI image.
 * Handles both 4-class output (glioma / meningioma / notumor / pituitary)
 * and the legacy 2-class output (normal / abnormal).
 *
 * Props:
 *   predictedClass  — string  e.g. "glioma" | "notumor" | "normal" | "abnormal"
 *   confidence      — float   e.g. 0.9412
 *   probabilities   — object  e.g. { glioma: 0.94, meningioma: 0.02, ... }
 *   modelUsed       — string  e.g. "efficientnet"
 */

// Classes that indicate a tumour is present
const TUMOR_CLASSES = new Set(['glioma', 'meningioma', 'pituitary', 'abnormal']);

const CLASS_LABELS = {
  glioma:      'Glioma',
  meningioma:  'Meningioma',
  pituitary:   'Pituitary Tumour',
  notumor:     'No Tumour',
  normal:      'Normal',
  abnormal:    'Abnormal',
};

// Human-readable display names for each architecture key
const MODEL_DISPLAY_NAMES = {
  mambavision: 'MambaVision-T',
  cnn:         'Lightweight CNN',
  vgg16:       'VGG-16',
  resnet50:    'ResNet-50',
  efficientnet:'EfficientNetB3',
};

const CLASS_COLORS = {
  glioma:     { bg: 'bg-red-50',    border: 'border-red-200',    text: 'text-red-700',    bar: 'bg-red-500',    icon: 'text-red-600'    },
  meningioma: { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700', bar: 'bg-orange-500', icon: 'text-orange-600' },
  pituitary:  { bg: 'bg-amber-50',  border: 'border-amber-200',  text: 'text-amber-700',  bar: 'bg-amber-500',  icon: 'text-amber-600'  },
  notumor:    { bg: 'bg-green-50',  border: 'border-green-200',  text: 'text-green-700',  bar: 'bg-green-500',  icon: 'text-green-600'  },
  normal:     { bg: 'bg-green-50',  border: 'border-green-200',  text: 'text-green-700',  bar: 'bg-green-500',  icon: 'text-green-600'  },
  abnormal:   { bg: 'bg-red-50',    border: 'border-red-200',    text: 'text-red-700',    bar: 'bg-red-500',    icon: 'text-red-600'    },
};

const DEFAULT_PALETTE = { bg: 'bg-pipeline-50', border: 'border-pipeline-200', text: 'text-pipeline-700', bar: 'bg-pipeline-500', icon: 'text-pipeline-600' };

export default function ResultCard({ predictedClass, confidence, probabilities, modelUsed }) {
  const key       = (predictedClass ?? '').toLowerCase();
  const isTumor   = TUMOR_CLASSES.has(key);
  const label     = CLASS_LABELS[key] ?? predictedClass ?? '—';
  const palette   = CLASS_COLORS[key] ?? DEFAULT_PALETTE;
  const pct       = confidence != null ? (confidence * 100).toFixed(1) : null;
  const modelLabel = modelUsed
    ? (MODEL_DISPLAY_NAMES[modelUsed.toLowerCase()] ?? modelUsed)
    : null;

  // Sort probabilities descending for display
  const probEntries = probabilities
    ? Object.entries(probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className={`card ${palette.bg} ${palette.border}`}>
      <div className="flex items-start gap-4">

        {/* Icon */}
        <div className={`mt-0.5 flex-shrink-0 ${palette.icon}`}>
          {isTumor ? (
            <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          ) : (
            <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">

          {/* Headline */}
          <p className={`text-2xl font-bold ${palette.text}`}>
            {isTumor ? 'Tumour Detected' : 'No Tumour Detected'}
          </p>
          <p className={`text-sm mt-1 ${palette.text} opacity-80`}>
            Predicted class: <span className="font-semibold">{label}</span>
            {modelLabel && (
              <span className="ml-2 text-xs opacity-60 font-mono">({modelLabel})</span>
            )}
          </p>

          {/* Confidence bar */}
          {pct != null && (
            <div className="mt-3">
              <div className="flex justify-between text-xs font-medium mb-1">
                <span className={palette.text}>Confidence</span>
                <span className={palette.text}>{pct}%</span>
              </div>
              <div className="h-2 rounded-full bg-pipeline-200 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${palette.bar}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )}

          {/* Per-class probability distribution */}
          {probEntries.length > 0 && (
            <div className="mt-4 space-y-1.5">
              <p className={`text-xs font-semibold uppercase tracking-wide ${palette.text} opacity-70`}>
                Class probabilities
              </p>
              {probEntries.map(([cls, prob]) => {
                const clsLabel = CLASS_LABELS[cls.toLowerCase()] ?? cls;
                const clsPct   = (prob * 100).toFixed(1);
                const isTop    = cls.toLowerCase() === key;
                return (
                  <div key={cls} className="flex items-center gap-2">
                    <span className={`w-28 text-xs truncate ${isTop ? `font-semibold ${palette.text}` : 'text-pipeline-500'}`}>
                      {clsLabel}
                    </span>
                    <div className="flex-1 h-1.5 rounded-full bg-pipeline-200 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${isTop ? palette.bar : 'bg-pipeline-400'}`}
                        style={{ width: `${clsPct}%` }}
                      />
                    </div>
                    <span className={`w-12 text-right text-xs font-mono ${isTop ? palette.text : 'text-pipeline-500'}`}>
                      {clsPct}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Clinical note */}
          <p className="mt-3 text-xs text-pipeline-500">
            {isTumor
              ? 'Please consult a qualified medical professional for further evaluation.'
              : 'No anomalies detected. Continue regular medical check-ups.'}
          </p>
        </div>

      </div>
    </div>
  );
}
