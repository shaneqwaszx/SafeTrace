import { Activity, AlertTriangle, Database, HardDrive, RefreshCcw } from 'lucide-react';
import { useState } from 'react';
import type { DashboardSummary, StorageSummary, SystemStatus } from '../types/analysis';
import { formatFileSize } from '../utils/formatters';

function Bars({ title, values }: { title: string; values: Record<string, number> }) {
  const maximum = Math.max(1, ...Object.values(values));
  return (
    <section className="border-t border-slate-200 pt-4">
      <h3 className="text-sm font-bold text-slate-950">{title}</h3>
      <div className="mt-3 space-y-2">
        {Object.entries(values).slice(0, 8).map(([label, value]) => (
          <div key={label} className="grid grid-cols-[minmax(100px,1fr)_2fr_auto] items-center gap-2 text-xs">
            <span className="truncate text-slate-600">{label}</span>
            <div className="h-2 bg-slate-100"><div className="h-2 bg-safety-blue" style={{ width: `${Math.max(3, value / maximum * 100)}%` }} /></div>
            <span className="font-bold text-slate-800">{value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function OperationalOverview({ dashboard, storage, system, onPreview, onApply }: {
  dashboard: DashboardSummary | null;
  storage: StorageSummary | null;
  system?: SystemStatus | null;
  onPreview: () => Promise<string>;
  onApply: () => Promise<string>;
}) {
  const [message, setMessage] = useState<string | null>(null);
  const cards = dashboard?.cards || {};
  const scheduler = system?.queue?.scheduler && typeof system.queue.scheduler === 'object'
    ? system.queue.scheduler as Record<string, unknown>
    : undefined;
  const adaptive = scheduler?.adaptiveWorkers && typeof scheduler.adaptiveWorkers === 'object'
    ? scheduler.adaptiveWorkers as Record<string, unknown>
    : undefined;
  const cardItems = [
    ['Active batches', cards.activeBatches ?? 0], ['Queued videos', cards.queuedVideos ?? 0],
    ['Running videos', cards.runningVideos ?? 0], ['Completed videos', cards.completedVideos ?? 0],
    ['Completed with failures', cards.completedWithFailures ?? 0], ['Violations found', cards.violationsFound ?? 0],
    ['Retrying / recovered', cards.retryingOrRecovered ?? 0], ['Storage used', formatFileSize(Number(cards.storageUsedBytes || 0))],
    ['Analysis workers', String(adaptive?.currentCapacity ?? system?.limits?.analysisConcurrency ?? 'unknown')],
    ['Worker admission', String(adaptive?.admissionReason || 'stable baseline')],
  ];
  return (
    <div className="space-y-5">
      <section className="bg-white p-5 shadow-soft ring-1 ring-slate-200">
        <div className="flex items-center gap-3"><Activity className="h-5 w-5 text-safety-teal" /><h2 className="text-lg font-bold">Operational overview</h2></div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{cardItems.map(([label, value]) => <div key={String(label)} className="min-w-0 border-l-2 border-safety-blue pl-3"><p className="text-xs font-bold uppercase text-slate-500">{label}</p><p className="mt-1 break-words text-base font-bold">{value}</p></div>)}</div>
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <Bars title="Jobs by status" values={dashboard?.jobsByStatus || {}} />
          <Bars title="Violations by type" values={dashboard?.violationsByType || {}} />
          <Bars title="Videos by vehicle / folder" values={dashboard?.videosBySourceGroup || {}} />
          <Bars title="Failures and retries by reason" values={dashboard?.failuresByReason || {}} />
        </div>
        <p className="mt-5 flex items-center gap-2 border-l-4 border-amber-400 bg-amber-50 p-3 text-sm text-amber-900"><AlertTriangle className="h-4 w-4" />{dashboard?.accuracyNotice || 'Operational metrics only. No accuracy claim is available.'}</p>
      </section>
      <section className="bg-white p-5 shadow-soft ring-1 ring-slate-200">
        <div className="flex items-center gap-3"><HardDrive className="h-5 w-5 text-safety-blue" /><h2 className="text-lg font-bold">Storage management</h2></div>
        <p className="mt-2 text-sm text-slate-600">Active, resumable, pinned, and exported items are protected from retention cleanup.</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{Object.entries(storage?.categories || {}).map(([name, bytes]) => <div key={name} className="border border-slate-200 p-3"><p className="text-xs font-bold uppercase text-slate-500">{name}</p><p className="mt-1 font-bold">{formatFileSize(bytes)}</p></div>)}</div>
        <div className="mt-4 flex flex-wrap gap-2">
          <button type="button" onClick={() => void onPreview().then(setMessage)} className="focus-ring inline-flex items-center gap-2 rounded border border-slate-300 px-3 py-2 text-sm font-semibold"><Database className="h-4 w-4" />Cleanup preview</button>
          <button type="button" onClick={() => { if (window.confirm('Delete only the expired items shown by a fresh cleanup preview?')) void onApply().then(setMessage); }} className="focus-ring inline-flex items-center gap-2 rounded border border-red-300 px-3 py-2 text-sm font-semibold text-red-700"><RefreshCcw className="h-4 w-4" />Apply confirmed cleanup</button>
        </div>
        {message ? <p className="mt-3 text-sm font-semibold text-safety-blue">{message}</p> : null}
      </section>
    </div>
  );
}
