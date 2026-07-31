import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

const COLORS = {
  CNN:          '#94a3b8',
  RFC:          '#64748b',
  ANN:          '#475569',
  'R-CNN':      '#334155',
  MambaVision:  '#2563eb',
};

const METRIC_LABELS = {
  accuracy:           'Accuracy (%)',
  sensitivity:        'Sensitivity (%)',
  specificity:        'Specificity (%)',
  psnr:               'PSNR (dB)',
  jaccard:            'Jaccard Index',
  computational_time: 'Compute Time (min)',
};

/**
 * ComparisonChart — bar chart comparing EDN-SVM vs baseline models.
 * @param {{ models: string[], metrics: object }} compareData
 */
export default function ComparisonChart({ compareData }) {
  if (!compareData?.models || !compareData?.metrics) {
    return (
      <div className="text-sm text-pipeline-400 text-center py-8">
        Comparison data not available
      </div>
    );
  }

  const { models, metrics } = compareData;

  // Build one chart per metric
  const metricKeys = Object.keys(metrics).filter((k) => k in METRIC_LABELS);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6">
      {metricKeys.map((metricKey) => {
        const values = metrics[metricKey];
        const chartData = models.map((model, idx) => ({
          model,
          value: values?.[idx] ?? 0,
        }));

        return (
          <div key={metricKey} className="card">
            <p className="section-title text-sm">{METRIC_LABELS[metricKey]}</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="model" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  formatter={(v) => [Number(v).toFixed(2), METRIC_LABELS[metricKey]]}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry) => (
                    <rect key={entry.model} fill={COLORS[entry.model] ?? '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}
