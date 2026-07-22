import { Archive, ChevronLeft, ChevronRight, Pin } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { deleteExport, exportJob, getBatchesPage, getCleanupHistory, getExportsPage, getJobsPage, pinJob } from '../services/analysisService';
import type { BatchStatus, DashboardSummary, ExportSummary, JobStatus, RecoverySummary, StorageSummary, SystemStatus } from '../types/analysis';
import { formatFileSize } from '../utils/formatters';
import { OperationalOverview } from './OperationalOverview';

const SECTIONS = ['Overview', 'Batches', 'Jobs', 'Evidence', 'Recovery', 'Storage', 'Exports', 'System'] as const;
type Section = typeof SECTIONS[number];
const PAGE_SIZE = 20;

export function OperationsWorkspace({ dashboard, storage, recovery, system, evidence, onPreview, onApply }: {
  dashboard: DashboardSummary | null;
  storage: StorageSummary | null;
  recovery: RecoverySummary | null;
  system: SystemStatus | null;
  evidence: ReactNode;
  onPreview: () => Promise<string>;
  onApply: () => Promise<string>;
}) {
  const [section, setSection] = useState<Section>('Overview');
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [batches, setBatches] = useState<BatchStatus[]>([]);
  const [exports, setExports] = useState<ExportSummary[]>([]);
  const [cleanupHistory, setCleanupHistory] = useState<Record<string, unknown>[]>([]);
  const [jobPage, setJobPage] = useState(1);
  const [batchPage, setBatchPage] = useState(1);
  const [exportPage, setExportPage] = useState(1);
  const [jobTotal, setJobTotal] = useState(0);
  const [batchTotal, setBatchTotal] = useState(0);
  const [exportTotal, setExportTotal] = useState(0);
  const [jobSearch, setJobSearch] = useState('');
  const [jobStatus, setJobStatus] = useState('all');
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [actionPending, setActionPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true); setError(null);
    try {
      const [jobResult, batchResult, exportResult, cleanupPage] = await Promise.all([
        getJobsPage(jobPage, PAGE_SIZE), getBatchesPage(batchPage, PAGE_SIZE), getExportsPage(exportPage, PAGE_SIZE), getCleanupHistory(),
      ]);
      setJobs(jobResult.items); setJobTotal(jobResult.total);
      setBatches(batchResult.items); setBatchTotal(batchResult.total);
      setExports(exportResult.items); setExportTotal(exportResult.total);
      setCleanupHistory(cleanupPage.items);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Operational data could not be loaded.');
    } finally { setLoading(false); }
  }
  useEffect(() => { void refresh(); }, [jobPage, batchPage, exportPage]);
  const visibleJobs = useMemo(() => jobs.filter((item) => {
    const haystack = `${item.jobId} ${String(item.sourceMetadata?.sourceRelativePath || '')}`.toLowerCase();
    return (jobStatus === 'all' || item.status === jobStatus) && haystack.includes(jobSearch.trim().toLowerCase());
  }), [jobs, jobSearch, jobStatus]);

  async function runSelected(action: 'pin' | 'export') {
    if (!selectedJobs.length || actionPending) return;
    setActionPending(true); setError(null);
    try {
      await Promise.all(selectedJobs.map((jobId) => action === 'pin' ? pinJob(jobId) : exportJob(jobId)));
      setSelectedJobs([]);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : `Selected ${action} action failed.`);
    } finally { setActionPending(false); }
  }

  return (
    <section className="space-y-4">
      <nav aria-label="Operations sections" className="flex flex-wrap gap-1 border-b border-slate-300">
        {SECTIONS.map((item) => <button type="button" key={item} onClick={() => setSection(item)} className={`focus-ring border-b-2 px-3 py-2 text-sm font-semibold ${section === item ? 'border-safety-blue text-safety-blue' : 'border-transparent text-slate-600'}`}>{item}</button>)}
      </nav>
      {loading ? <p role="status" className="rounded border border-slate-200 bg-white p-4">Loading backend operational state...</p> : null}
      {error ? <div className="rounded border border-red-200 bg-red-50 p-4 text-red-900"><p>{error}</p><button type="button" onClick={() => void refresh()} className="mt-2 font-semibold underline">Retry</button></div> : null}
      {section === 'Overview' ? <OperationalOverview dashboard={dashboard} storage={storage} system={system} onPreview={onPreview} onApply={onApply} /> : null}
      {section === 'Evidence' ? evidence : null}
      {section === 'Batches' ? <div className="space-y-3"><BatchRows batches={batches} /><PageControls label="Batch pages" page={batchPage} total={batchTotal} onPage={setBatchPage} /></div> : null}
      {section === 'Jobs' ? <div className="space-y-3"><div className="grid gap-2 bg-white p-4 shadow-soft sm:grid-cols-[1fr_12rem_auto]"><label className="text-sm font-semibold">Search jobs<input aria-label="Search jobs" value={jobSearch} onChange={(event) => setJobSearch(event.target.value)} className="mt-1 w-full rounded border border-slate-300 px-3 py-2 font-normal" /></label><label className="text-sm font-semibold">Status<select aria-label="Filter jobs by status" value={jobStatus} onChange={(event) => setJobStatus(event.target.value)} className="mt-1 w-full rounded border border-slate-300 px-3 py-2 font-normal"><option value="all">All statuses</option>{['queued', 'running', 'retry_wait', 'recovering', 'paused', 'completed', 'failed', 'cancelled'].map((status) => <option key={status}>{status}</option>)}</select></label><div className="flex items-end gap-2"><button type="button" disabled={!selectedJobs.length || actionPending} onClick={() => void runSelected('pin')} className="focus-ring inline-flex items-center gap-1 rounded border border-slate-300 px-3 py-2 text-sm font-semibold disabled:opacity-50"><Pin className="h-4 w-4" />Pin</button><button type="button" disabled={!selectedJobs.length || actionPending} onClick={() => void runSelected('export')} className="focus-ring inline-flex items-center gap-1 rounded border border-safety-blue px-3 py-2 text-sm font-semibold text-safety-blue disabled:opacity-50"><Archive className="h-4 w-4" />Export</button></div></div><JobRows jobs={visibleJobs} selected={selectedJobs} onSelected={setSelectedJobs} /><PageControls label="Job pages" page={jobPage} total={jobTotal} onPage={setJobPage} /></div> : null}
      {section === 'Recovery' ? <SimpleRows title="Recovery" empty="No interrupted work requires a decision." rows={(recovery?.jobs || []).map((item) => ({ id: item.jobId, title: item.sourceRelativePath, detail: `${item.lastStage} - ${item.progressPercent}% retained - ${formatFileSize(item.storedBytes)} - ${item.resumableReason}` }))} /> : null}
      {section === 'Storage' ? <div className="space-y-4"><OperationalOverview dashboard={dashboard} storage={storage} onPreview={onPreview} onApply={onApply} /><SimpleRows title="Cleanup history" empty="No retention cleanup history." rows={cleanupHistory.map((item, index) => ({ id: String(item.auditPath || index), title: String(item.appliedAt || 'Cleanup run'), detail: `${formatFileSize(Number(item.actualReclaimedBytes || 0))} reclaimed` }))} /></div> : null}
      {section === 'Exports' ? <div className="space-y-3"><div className="bg-white p-5 shadow-soft"><h2 className="font-bold">Verified exports</h2>{exports.length ? <div className="mt-3 space-y-2">{exports.map((item) => <div key={item.exportId} className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 py-3"><div><p className="font-semibold">{item.exportId}</p><p className="text-sm text-slate-600">{item.status} - {formatFileSize(item.sizeBytes)} - {item.createdAt}</p></div><button type="button" onClick={() => { if (window.confirm('Delete this verified export?')) void deleteExport(item.exportId).then(refresh).catch((reason) => setError(reason instanceof Error ? reason.message : 'Delete failed')); }} className="focus-ring rounded border border-red-300 px-3 py-2 text-sm font-semibold text-red-700">Delete export</button></div>)}</div> : <p className="mt-3 text-sm text-slate-600">No verified exports.</p>}</div><PageControls label="Export pages" page={exportPage} total={exportTotal} onPage={setExportPage} /></div> : null}
      {section === 'System' ? <div className="bg-white p-5 shadow-soft"><h2 className="font-bold">System</h2><p className="mt-2 text-sm text-slate-600">Backend {system?.backend_version || 'not connected'} - device {system?.device || 'unknown'} - GPU {system?.gpuAvailable ? 'available' : 'not available'}.</p><details className="mt-3"><summary className="cursor-pointer font-semibold">Backend details</summary><pre className="mt-2 max-h-96 overflow-auto text-xs">{JSON.stringify(system, null, 2)}</pre></details></div> : null}
    </section>
  );
}

function SimpleRows({ title, empty, rows }: { title: string; empty: string; rows: Array<{ id: string; title: string; detail: string }> }) {
  return <div className="bg-white p-5 shadow-soft"><h2 className="font-bold">{title}</h2>{rows.length ? <div className="mt-3 divide-y divide-slate-200">{rows.map((row) => <div key={row.id} className="py-3"><p className="font-semibold">{row.title}</p><p className="mt-1 text-sm text-slate-600">{row.detail}</p></div>)}</div> : <p className="mt-3 text-sm text-slate-600">{empty}</p>}</div>;
}

function JobRows({ jobs, selected, onSelected }: { jobs: JobStatus[]; selected: string[]; onSelected: (ids: string[]) => void }) {
  return <div className="bg-white p-5 shadow-soft"><h2 className="font-bold">Jobs</h2>{jobs.length ? <div className="mt-3 divide-y divide-slate-200">{jobs.map((item) => <label key={item.jobId} className="flex gap-3 py-3"><input type="checkbox" aria-label={`Select ${item.jobId}`} checked={selected.includes(item.jobId)} onChange={(event) => onSelected(event.target.checked ? [...selected, item.jobId] : selected.filter((id) => id !== item.jobId))} /><span><span className="block font-semibold">{String(item.sourceMetadata?.sourceRelativePath || item.jobId)}</span><span className="mt-1 block text-sm text-slate-600">{item.status} - {item.progressPercent}% - {item.requestedModeLabel || 'Fast Local Analysis'} - {item.actualDeviceLabel || 'device pending'} - queue {formatRuntime(item.queueWaitSeconds)} - processing {formatRuntime(item.analysisRuntimeSeconds)} - total {formatRuntime(item.elapsedSeconds)} - updated {item.updatedAt}</span></span></label>)}</div> : <p className="mt-3 text-sm text-slate-600">No backend jobs.</p>}</div>;
}

function formatRuntime(value?: number | null) {
  if (typeof value !== 'number') return 'pending';
  const seconds = Math.max(0, Math.floor(value));
  const minutes = Math.floor(seconds / 60);
  return minutes ? `${minutes}m ${String(seconds % 60).padStart(2, '0')}s` : `${seconds}s`;
}

function BatchRows({ batches }: { batches: BatchStatus[] }) {
  return <div className="bg-white p-5 shadow-soft"><h2 className="font-bold">Batches</h2>{batches.length ? <div className="mt-3 divide-y divide-slate-200">{batches.map((item) => <details key={item.batchId} className="py-3"><summary className="cursor-pointer font-semibold">{item.sourceFilename || item.batchId}</summary><p className="mt-1 text-sm text-slate-600">{item.status} - {item.jobIds.length} jobs - updated {item.updatedAt}</p><pre className="mt-2 max-h-48 overflow-auto rounded bg-slate-50 p-2 text-xs">{JSON.stringify(item.hierarchy || {}, null, 2)}</pre></details>)}</div> : <p className="mt-3 text-sm text-slate-600">No backend batches.</p>}</div>;
}

function PageControls({ label, page, total, onPage }: { label: string; page: number; total: number; onPage: (page: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  return <nav aria-label={label} className="flex items-center justify-between bg-white px-4 py-2 text-sm shadow-soft"><button type="button" title="Previous page" disabled={page <= 1} onClick={() => onPage(Math.max(1, page - 1))} className="focus-ring rounded p-2 disabled:opacity-40"><ChevronLeft className="h-4 w-4" /></button><span>Page {page} of {pages} ({total} total)</span><button type="button" title="Next page" disabled={page >= pages} onClick={() => onPage(Math.min(pages, page + 1))} className="focus-ring rounded p-2 disabled:opacity-40"><ChevronRight className="h-4 w-4" /></button></nav>;
}
