import { Archive, Pin, PinOff, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { exportAndDeleteJob, exportJob, pinJob, unpinJob } from '../services/analysisService';

export function ResultLifecycleActions({ jobId, onDeleted }: { jobId?: string | null; onDeleted: () => void }) {
  const [pending, setPending] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  if (!jobId) return null;

  async function run(name: string, action: () => Promise<unknown>, success: string) {
    if (pending) return;
    setPending(name);
    setMessage(null);
    try {
      await action();
      setMessage(success);
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : `${name} failed`);
    } finally {
      setPending(null);
    }
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-soft">
      <h2 className="text-sm font-bold text-slate-950">Result lifecycle</h2>
      <p className="mt-1 text-sm text-slate-600">Pin retention, create a verified backend export, or export and remove backend working data.</p>
      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" disabled={Boolean(pending)} onClick={() => void run('Pin', () => pinJob(jobId), 'Result pinned.')} className="focus-ring inline-flex items-center gap-2 rounded border border-slate-300 px-3 py-2 text-sm font-semibold disabled:opacity-50"><Pin className="h-4 w-4" />Pin result</button>
        <button type="button" disabled={Boolean(pending)} onClick={() => void run('Unpin', () => unpinJob(jobId), 'Result unpinned.')} className="focus-ring inline-flex items-center gap-2 rounded border border-slate-300 px-3 py-2 text-sm font-semibold disabled:opacity-50"><PinOff className="h-4 w-4" />Unpin result</button>
        <button type="button" disabled={Boolean(pending)} onClick={() => void run('Export', () => exportJob(jobId), 'Verified backend export created.')} className="focus-ring inline-flex items-center gap-2 rounded border border-safety-blue px-3 py-2 text-sm font-semibold text-safety-blue disabled:opacity-50"><Archive className="h-4 w-4" />Export result</button>
        <button type="button" disabled={Boolean(pending)} onClick={() => { if (window.confirm('Create and verify an export, then delete this backend job and its working data?')) void run('Export and delete', () => exportAndDeleteJob(jobId), 'Export verified and backend working data deleted.').then(onDeleted); }} className="focus-ring inline-flex items-center gap-2 rounded border border-red-300 px-3 py-2 text-sm font-semibold text-red-700 disabled:opacity-50"><Trash2 className="h-4 w-4" />Export, verify, then delete</button>
      </div>
      {pending ? <p className="mt-3 text-sm text-slate-600" role="status">{pending} in progress...</p> : null}
      {message ? <p className="mt-3 text-sm font-semibold text-safety-blue" role="status">{message}</p> : null}
    </section>
  );
}
