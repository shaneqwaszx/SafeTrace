import { AlertTriangle, CheckCircle2, FileJson, Gauge, ShieldCheck } from 'lucide-react';
import { useMemo, useState } from 'react';

type BenchmarkRow = {
  sampleId?: string;
  profile?: string;
  status?: string;
  runtimeSeconds?: number;
  error?: string | null;
};

type BenchmarkReport = {
  runId?: string;
  mode?: string;
  device?: string;
  concurrency?: number;
  configuration?: { checkpoints?: Array<{ name?: string; path?: string; sha256?: string }> };
  accuracyMetricsAvailable?: boolean;
  accuracyGatePassed?: boolean;
  summary?: Record<string, number | string | null>;
  rows?: BenchmarkRow[];
};

function value(report: BenchmarkReport | null, key: string): string {
  const item = report?.summary?.[key];
  if (item === undefined || item === null) return 'Not reported';
  return typeof item === 'number' ? item.toFixed(item % 1 ? 2 : 0) : String(item);
}

function ReportLoader({ label, onLoad }: { label: string; onLoad: (value: BenchmarkReport) => void }) {
  const [error, setError] = useState<string | null>(null);
  return (
    <label className="focus-within:ring-2 focus-within:ring-safety-blue flex cursor-pointer items-center gap-3 rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-800">
      <FileJson className="h-5 w-5 text-safety-blue" aria-hidden="true" />
      <span>{label}</span>
      <input
        className="sr-only"
        type="file"
        accept="application/json,.json"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          try {
            onLoad(JSON.parse(await file.text()) as BenchmarkReport);
            setError(null);
          } catch (reason) {
            setError(reason instanceof Error ? reason.message : 'Invalid benchmark JSON');
          }
        }}
      />
      {error ? <span className="text-xs text-red-700">{error}</span> : null}
    </label>
  );
}

function ReportSummary({ title, report }: { title: string; report: BenchmarkReport | null }) {
  const failed = report?.rows?.filter((row) => row.status !== 'completed' && row.status !== 'dry_run') ?? [];
  return (
    <section className="border-t border-slate-200 py-5 first:border-t-0">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-950">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">Run {report?.runId || 'not loaded'}</p>
        </div>
        <span className={`rounded border px-2.5 py-1 text-xs font-bold uppercase ${report?.accuracyGatePassed ? 'border-emerald-300 bg-emerald-50 text-emerald-800' : 'border-amber-300 bg-amber-50 text-amber-800'}`}>
          {report?.accuracyGatePassed ? 'Accuracy gate passed' : 'Accuracy gate not passed'}
        </span>
      </div>
      <dl className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Mode', report?.mode || 'Not reported'],
          ['Device', report?.device || 'Not reported'],
          ['Concurrency', report?.concurrency ?? 'Not reported'],
          ['Completed items', value(report, 'completedItems')],
          ['Mean runtime', `${value(report, 'meanRuntimeSeconds')} s`],
          ['P95 runtime', `${value(report, 'p95RuntimeSeconds')} s`],
          ['Frames screened', value(report, 'framesScreened')],
          ['Checkpoint', report?.configuration?.checkpoints?.find((item) => item.name === 'fallbackDetector')?.path || 'Not reported'],
        ].map(([label, metric]) => (
          <div key={String(label)} className="min-w-0 border-l-2 border-slate-200 pl-3">
            <dt className="text-xs font-bold uppercase text-slate-500">{label}</dt>
            <dd className="mt-1 break-words text-sm font-semibold text-slate-900">{metric}</dd>
          </div>
        ))}
      </dl>
      {failed.length ? (
        <div className="mt-4 border-l-4 border-red-400 bg-red-50 p-3 text-sm text-red-900">
          <p className="font-bold">Failed items ({failed.length})</p>
          <ul className="mt-1 space-y-1">
            {failed.slice(0, 10).map((row, index) => <li key={`${row.sampleId}-${index}`}>{row.sampleId || 'Unknown item'}: {row.error || row.status}</li>)}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export function ValidationDashboard() {
  const [baseline, setBaseline] = useState<BenchmarkReport | null>(null);
  const [candidate, setCandidate] = useState<BenchmarkReport | null>(null);
  const delta = useMemo(() => {
    const before = Number(baseline?.summary?.meanRuntimeSeconds);
    const after = Number(candidate?.summary?.meanRuntimeSeconds);
    return Number.isFinite(before) && Number.isFinite(after) ? after - before : null;
  }, [baseline, candidate]);

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-6 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <header className="border-b border-slate-300 pb-5">
          <p className="text-sm font-bold text-safety-teal">Internal validation</p>
          <h1 className="mt-1 text-3xl font-bold tracking-normal">Detector readiness</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">Review reproducible benchmark and soak artifacts before promoting a detector configuration.</p>
        </header>

        <section className="mt-5 border-l-4 border-amber-400 bg-amber-50 p-4">
          <div className="flex gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" aria-hidden="true" />
            <div>
              <p className="font-bold text-amber-950">Labelled accuracy is a release gate</p>
              <p className="mt-1 text-sm leading-6 text-amber-900">Unlabelled footage can validate runtime, schema, and stability only. It must not be presented as precision, recall, or F1 evidence.</p>
            </div>
          </div>
        </section>

        <div className="mt-5 flex flex-wrap gap-3">
          <ReportLoader label="Load baseline JSON" onLoad={setBaseline} />
          <ReportLoader label="Load candidate JSON" onLoad={setCandidate} />
        </div>

        <section className="mt-6 bg-white px-5 shadow-sm ring-1 ring-slate-200">
          <ReportSummary title="Current baseline" report={baseline} />
          <ReportSummary title="Candidate" report={candidate} />
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-3">
          <div className="border border-slate-200 bg-white p-4">
            <Gauge className="h-5 w-5 text-safety-blue" aria-hidden="true" />
            <p className="mt-3 text-xs font-bold uppercase text-slate-500">Mean runtime change</p>
            <p className="mt-1 text-xl font-bold">{delta === null ? 'Not comparable' : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} s`}</p>
          </div>
          <div className="border border-slate-200 bg-white p-4">
            <ShieldCheck className="h-5 w-5 text-safety-teal" aria-hidden="true" />
            <p className="mt-3 text-xs font-bold uppercase text-slate-500">Accuracy evidence</p>
            <p className="mt-1 text-xl font-bold">{candidate?.accuracyMetricsAvailable ? 'Labelled metrics present' : 'Insufficient labels'}</p>
          </div>
          <div className="border border-slate-200 bg-white p-4">
            <CheckCircle2 className="h-5 w-5 text-emerald-700" aria-hidden="true" />
            <p className="mt-3 text-xs font-bold uppercase text-slate-500">Promotion status</p>
            <p className="mt-1 text-xl font-bold">{candidate?.accuracyGatePassed ? 'Eligible for review' : 'Do not promote'}</p>
          </div>
        </section>
      </div>
    </main>
  );
}
