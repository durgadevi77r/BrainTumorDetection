/**
 * MetricsTable — displays global model evaluation metrics.
 *
 * Metrics are sourced from GET /api/metrics which reads from the
 * model_metrics table (populated by POST /api/evaluate).
 *
 * Value formatting:
 *  - accuracy / sensitivity / specificity — already in % from backend
 *  - psnr                                 — in dB
 *  - jaccard / ber                        — raw ratio [0–1]
 *  - computational_time                   — in ms
 *
 * When a value is null/undefined the cell shows '—'.
 * When metrics.status is present, the data is a placeholder and a
 * warning banner is shown above the table.
 */

const METRIC_DEFS = [
  { key: 'accuracy',           label: 'Accuracy',       eq: 'Eq. 28', unit: '%',  target: 98.51, decimals: 2 },
  { key: 'sensitivity',        label: 'Sensitivity',    eq: 'Eq. 31', unit: '%',  target: 98.2,  decimals: 2 },
  { key: 'specificity',        label: 'Specificity',    eq: 'Eq. 32', unit: '%',  target: 98.9,  decimals: 2 },
  { key: 'psnr',               label: 'PSNR',           eq: 'Eq. 30', unit: 'dB', target: 52.98, decimals: 2 },
  { key: 'jaccard',            label: 'Jaccard Index',  eq: 'Eq. 29', unit: '',   target: null,  decimals: 4 },
  { key: 'ber',                label: 'BER',            eq: '—',      unit: '',   target: null,  decimals: 4 },
  { key: 'computational_time', label: 'Compute Time',   eq: '—',      unit: 'ms', target: null,  decimals: 2 },
];

export default function MetricsTable({ metrics }) {
  if (!metrics) {
    return (
      <div className="text-sm text-pipeline-400 text-center py-8">
        Metrics not yet computed.{' '}
        <span className="text-blue-500">
          Run <code className="font-mono text-xs">POST /api/evaluate</code> to generate real values.
        </span>
      </div>
    );
  }

  const isPlaceholder = !!metrics.status;

  return (
    <div className="space-y-2">
      {isPlaceholder && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
          Showing paper-target placeholders — no evaluation has been run yet.
          Run <code className="font-mono">POST /api/evaluate</code> to populate real values.
        </div>
      )}

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
            {METRIC_DEFS.map(({ key, label, eq, unit, target, decimals }) => {
              const val = metrics[key];
              const hasValue = val != null && !isNaN(Number(val));
              const displayVal = hasValue
                ? `${Number(val).toFixed(decimals)}${unit ? ' ' + unit : ''}`
                : 'N/A';

              return (
                <tr key={key} className="hover:bg-pipeline-50 transition-colors">
                  <td className="px-4 py-3 font-medium text-pipeline-800">{label}</td>
                  <td className="px-4 py-3 text-blue-600 font-mono text-xs hidden sm:table-cell">{eq}</td>
                  <td className={`px-4 py-3 text-right font-mono ${hasValue ? 'text-pipeline-700' : 'text-pipeline-400'}`}>
                    {displayVal}
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
    </div>
  );
}
