import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Clock3, Play, RefreshCw, Trash2 } from 'lucide-react';
import type { RecoverySummary } from '../types/analysis';
import { formatFileSize } from '../utils/formatters';

export function RecoveryBanner({ recovery, busy, onContinue, onRestartFresh, onDiscard, onLater }: {
  recovery: RecoverySummary;
  busy: boolean;
  onContinue: (jobIds: string[]) => void;
  onRestartFresh: (jobIds: string[], purgeCompatibleCache: boolean) => void;
  onDiscard: (jobIds: string[]) => void;
  onLater: (jobIds: string[]) => void;
}) {
  const candidateIds = useMemo(() => recovery.jobs.map((job) => job.jobId), [recovery.jobs]);
  const [selectedIds, setSelectedIds] = useState<string[]>(candidateIds);
  const [purgeCompatibleCache, setPurgeCompatibleCache] = useState(false);

  useEffect(() => {
    setSelectedIds((current) => {
      const retained = current.filter((jobId) => candidateIds.includes(jobId));
      return retained.length ? retained : candidateIds;
    });
  }, [candidateIds]);

  if (!recovery.candidateCount) return null;
  const selected = new Set(selectedIds);
  const toggleJob = (jobId: string) => {
    setSelectedIds((current) => current.includes(jobId)
      ? current.filter((value) => value !== jobId)
      : [...current, jobId]);
  };
  const noSelection = selectedIds.length === 0;

  return (
    <section className="border-l-4 border-amber-500 bg-amber-50 p-4 text-amber-950" role="status">
      <div className="flex gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h2 className="font-bold">Interrupted work is ready for recovery</h2>
          <p className="mt-1 text-sm leading-6">{recovery.candidateCount} job{recovery.candidateCount === 1 ? '' : 's'} using {formatFileSize(recovery.storedBytes)} were preserved. Completed results remain available.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" disabled={busy || noSelection} onClick={() => onContinue(selectedIds)} className="focus-ring inline-flex items-center gap-2 rounded bg-safety-blue px-3 py-2 text-sm font-semibold text-white disabled:opacity-60"><Play className="h-4 w-4" />Continue</button>
            <button type="button" disabled={busy || noSelection} onClick={() => onRestartFresh(selectedIds, purgeCompatibleCache)} className="focus-ring inline-flex items-center gap-2 rounded border border-safety-blue bg-white px-3 py-2 text-sm font-semibold text-safety-blue disabled:opacity-60"><RefreshCw className="h-4 w-4" />Restart fresh</button>
            <button type="button" disabled={busy || noSelection} onClick={() => onDiscard(selectedIds)} className="focus-ring inline-flex items-center gap-2 rounded border border-red-300 bg-white px-3 py-2 text-sm font-semibold text-red-700 disabled:opacity-60"><Trash2 className="h-4 w-4" />Discard</button>
            <button type="button" disabled={busy || noSelection} onClick={() => onLater(selectedIds)} className="focus-ring inline-flex items-center gap-2 rounded border border-amber-400 bg-white px-3 py-2 text-sm font-semibold disabled:opacity-60"><Clock3 className="h-4 w-4" />Later</button>
          </div>
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer font-semibold">Recovery details</summary>
            <div className="mt-2 flex flex-wrap gap-3">
              <button type="button" className="focus-ring font-semibold text-safety-blue" onClick={() => setSelectedIds(candidateIds)}>Select all unfinished</button>
              <button type="button" className="focus-ring font-semibold text-safety-blue" onClick={() => setSelectedIds([])}>Clear selection</button>
            </div>
            <ul className="mt-2 space-y-2">
              {recovery.jobs.map((job) => (
                <li key={job.jobId}>
                  <label className="flex cursor-pointer items-start gap-2">
                    <input type="checkbox" className="mt-1" checked={selected.has(job.jobId)} onChange={() => toggleJob(job.jobId)} />
                    <span><span className="font-semibold">{job.sourceRelativePath}</span> - {job.lastStage} ({job.progressPercent}%)</span>
                  </label>
                </li>
              ))}
            </ul>
            <label className="mt-3 flex items-start gap-2 rounded border border-amber-300 bg-white p-3">
              <input type="checkbox" className="mt-1" checked={purgeCompatibleCache} onChange={(event) => setPurgeCompatibleCache(event.target.checked)} />
              <span><span className="font-semibold">Also purge compatible reusable cache for these source files</span><br /><span className="text-xs">Off by default. A separate confirmation is required.</span></span>
            </label>
          </details>
        </div>
      </div>
    </section>
  );
}
