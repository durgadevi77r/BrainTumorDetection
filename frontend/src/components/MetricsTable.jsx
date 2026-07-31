const METRIC_DEFS = [
  { key: 'accuracy',           label: 'Accuracy',       eq: 'Eq. 28', unit: '%',  target: 98.51 },
  { key: 'sensitivity',        label: 'Sensitivity',    eq: 'Eq. 31', unit: '%',  target: 98.2  },
  { key: 'specificity',        label: 'Specificity',    eq: 'Eq. 32', unit: '%',  target: 98.9  },
  { key: 'psnr',               label: 'PSNR',           eq: 'Eq. 30', unit: 'dB', target: 52.98 },
  { key: 'jaccard',            label: 'Jaccard Index',  eq: 'Eq. 29', unit: '',   target: null  },
  { key: 'ber',                label: 'BER',            eq: '—',      unit: '',   target: null  },
  { key: 'computational_time', label: 'Compute Time',   eq: '—',      unit: 'ms', target: null  },
];

/**
 * MetricsTable — displays per-image evaluation metrics.
 * @param {object} metrics
 */
export default function MetricsTable({ metrics }) {
  if (!metrics) {
    return (
      <div className="text-sm text-pipeline-400 text-center py-8">
        Metrics not yet computed
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-pipeline-200">
      <table className="min-w-full divide-y divide-pipeline-100 text-sm">
        <thead className="bg-pipeline-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-semibold text-pipeline-500 uppercase tracking-wide">Metric</th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-pipeline-500 uppercase tracking-wide hidden sm:table-cell">Eq.</th>
            <th className="px-4 py-3 text-right text-xs font-semibold text-pipeline-500 uppercase tracking-wide">Value</th>
            <th className="px-4 py-3 text-right text-xs font-semibold text-pipeline-500 uppercase tracking-wide hidden sm:table-cell">Paper Target</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-pipeline-100 bg-white">
          {METRIC_DEFS.map(({ key, label, eq, unit, target }) => {
            const val = metrics[key];
            return (
              <tr key={key} className="hover:bg-pipeline-50 transition-colors">
                <td className="px-4 py-3 font-medium text-pipeline-800">{label}</td>
                <td className="px-4 py-3 text-blue-600 font-mono text-xs hidden sm:table-cell">{eq}</td>
                <td className="px-4 py-3 text-right font-mono text-pipeline-700">
                  {val != null ? `${Number(val).toFixed(2)} ${unit}` : '—'}
                </td>
                <td className="px-4 py-3 text-right text-pipeline-400 text-xs hidden sm:table-cell">
                  {target != null ? `${target} ${unit}` : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
