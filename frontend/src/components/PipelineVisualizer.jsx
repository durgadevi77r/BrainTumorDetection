const STEPS = [
  { key: 'upload',     label: 'Upload',        desc: 'T1-mode image input' },
  { key: 'resize',     label: 'Resize',        desc: '256 × 256 px' },
  { key: 'preprocess', label: 'ACEA',          desc: 'Contrast enhancement' },
  { key: 'denoise',    label: 'Median Filter', desc: 'Noise removal' },
  { key: 'segment',    label: 'FCM',           desc: 'Segmentation (C=3)' },
  { key: 'features',   label: 'GLCM',          desc: '7 texture features' },
  { key: 'classify',   label: 'MambaVision',   desc: 'Classification' },
];

/**
 * PipelineVisualizer — horizontal step tracker.
 * @param {string} activeStep — key of the currently active step
 * @param {Set<string>} completedSteps — set of completed step keys
 */
export default function PipelineVisualizer({ activeStep = null, completedSteps = new Set() }) {
  return (
    <div className="w-full overflow-x-auto">
      <ol className="flex items-start gap-0 min-w-max">
        {STEPS.map((step, idx) => {
          const done   = completedSteps.has(step.key);
          const active = activeStep === step.key;
          const isLast = idx === STEPS.length - 1;

          return (
            <li key={step.key} className="flex items-start">
              {/* Step node */}
              <div className="flex flex-col items-center">
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-all duration-300
                    ${done
                      ? 'bg-blue-600 border-blue-600 text-white'
                      : active
                        ? 'bg-white border-blue-500 text-blue-600 shadow-sm'
                        : 'bg-white border-pipeline-200 text-pipeline-400'
                    }`}
                >
                  {done ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5} aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  ) : (
                    idx + 1
                  )}
                </div>
                <p className={`mt-2 text-xs font-semibold text-center w-20 ${active ? 'text-blue-600' : done ? 'text-pipeline-700' : 'text-pipeline-400'}`}>
                  {step.label}
                </p>
                <p className="text-xs text-pipeline-400 text-center w-24 leading-tight hidden sm:block">
                  {step.desc}
                </p>
              </div>

              {/* Connector line */}
              {!isLast && (
                <div className={`h-0.5 w-10 mt-4 mx-1 flex-shrink-0 transition-colors duration-300 ${done ? 'bg-blue-500' : 'bg-pipeline-200'}`} />
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
