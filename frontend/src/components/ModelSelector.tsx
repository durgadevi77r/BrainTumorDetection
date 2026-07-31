/**
 * ModelSelector — dropdown for choosing an architecture.
 */

import type { ArchitectureName, ModelInfo } from '@/types';

const ARCH_LABELS: Record<string, string> = {
  mambavision: 'MambaVision-T',
  cnn:         'Lightweight CNN',
  vgg16:       'VGG-16',
  resnet50:    'ResNet-50',
  efficientnet:'EfficientNetB3',
};

interface Props {
  value: ArchitectureName | string;
  onChange: (name: ArchitectureName) => void;
  models?: ModelInfo[];
  disabled?: boolean;
  label?: string;
  id?: string;
}

const ALL_ARCHS: ArchitectureName[] = ['mambavision', 'cnn', 'vgg16', 'resnet50', 'efficientnet'];

export default function ModelSelector({ value, onChange, models, disabled, label = 'Model', id = 'model-select' }: Props) {
  const available = new Set(models?.filter((m) => m.available).map((m) => m.name) ?? ALL_ARCHS);

  return (
    <div className="space-y-1.5" data-testid="model-selector">
      <label htmlFor={id} className="block text-xs font-semibold text-pipeline-600 uppercase tracking-wide">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as ArchitectureName)}
        disabled={disabled}
        className="w-full rounded-lg border border-pipeline-200 bg-white px-3 py-2.5 text-sm text-pipeline-800
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
          disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {ALL_ARCHS.map((arch) => (
          <option key={arch} value={arch} disabled={!available.has(arch)}>
            {ARCH_LABELS[arch] ?? arch}
            {!available.has(arch) ? ' (not trained)' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
