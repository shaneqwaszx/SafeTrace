import type { FrameResult, UseCaseProfileSelection } from '../types/analysis';
import { AlertTriangle, Download, Grid2X2, List } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { FrameEvidenceCard } from './FrameEvidenceCard';

type EvidenceFramesProps = {
  frames: FrameResult[];
  showExplanations: boolean;
  highlightedFrameId?: string | null;
  jobId?: string | null;
  useCaseProfile?: UseCaseProfileSelection;
  effectiveQuery?: string;
  analysisDiagnostics?: Record<string, unknown> | null;
  acceptedFindingCount?: number;
  evidenceStatus?: string;
};

export function EvidenceFrames({
  frames,
  showExplanations,
  highlightedFrameId,
  jobId,
  useCaseProfile,
  effectiveQuery,
  analysisDiagnostics,
  acceptedFindingCount = 0,
  evidenceStatus,
}: EvidenceFramesProps) {
  const [findingFilter, setFindingFilter] = useState('all');
  const [sortMode, setSortMode] = useState<'timestamp' | 'confidence'>('timestamp');
  const [viewMode, setViewMode] = useState<'list' | 'gallery'>('list');
  const ownedFrames = useMemo(
    () => frames.filter((frame) => !jobId || frame.jobId === jobId),
    [frames, jobId],
  );
  const canonicalFrames = useMemo(() => [...ownedFrames].sort((left, right) => {
    const pathDelta = String(left.sourceRelativePath || left.videoFilename || '').localeCompare(
      String(right.sourceRelativePath || right.videoFilename || ''),
      undefined,
      { numeric: true, sensitivity: 'base' },
    );
    if (pathDelta) return pathDelta;
    const timestampDelta = (left.timestampSeconds ?? 0) - (right.timestampSeconds ?? 0);
    if (timestampDelta) return timestampDelta;
    const sourceDelta = (left.sourceFrameIndex ?? Number.MAX_SAFE_INTEGER) - (right.sourceFrameIndex ?? Number.MAX_SAFE_INTEGER);
    if (sourceDelta) return sourceDelta;
    return String(left.evidenceId || left.id).localeCompare(String(right.evidenceId || right.id), undefined, { numeric: true });
  }), [ownedFrames]);
  const presentationNumbers = useMemo(
    () => new Map(canonicalFrames.map((frame, index) => [frame.evidenceId || frame.id, index + 1])),
    [canonicalFrames],
  );
  const foreignFrameCount = frames.length - ownedFrames.length;
  const findings = useMemo(() => Array.from(new Set(canonicalFrames.flatMap((frame) => frame.violations.map((item) => item.name)))).sort(), [canonicalFrames]);
  useEffect(() => {
    setFindingFilter('all');
    setSortMode('timestamp');
  }, [jobId]);
  const visibleFrames = useMemo(() => {
    const filtered = findingFilter === 'all' ? canonicalFrames : canonicalFrames.filter((frame) => frame.violations.some((item) => item.name === findingFilter));
    return [...filtered].sort((left, right) => {
      if (sortMode === 'confidence') {
        return Math.max(0, ...right.violations.map((item) => item.confidence)) - Math.max(0, ...left.violations.map((item) => item.confidence));
      }
      return (left.timestampSeconds ?? left.frameNumber) - (right.timestampSeconds ?? right.frameNumber);
    });
  }, [canonicalFrames, findingFilter, sortMode]);

  function exportEvidenceMetadata() {
    const payload = visibleFrames.map((frame) => ({
      frameId: frame.id,
      sourceRelativePath: frame.sourceRelativePath,
      videoFilename: frame.videoFilename,
      timestamp: frame.timestampLabel || frame.timestamp,
      timestampSeconds: frame.timestampSeconds,
      frameNumber: frame.frameNumber,
      presentationFrameNumber: presentationNumbers.get(frame.evidenceId || frame.id),
      sourceFrameIndex: frame.sourceFrameIndex,
      findings: frame.violations.map((item) => ({ name: item.name, confidence: item.confidence, severity: item.severity })),
      actualAnalysisMode: frame.explanationSource,
    }));
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'safetrace-evidence-metadata.json';
    link.click();
    URL.revokeObjectURL(url);
  }

  if (acceptedFindingCount > 0 && evidenceStatus === 'unavailable') {
    return (
      <section id="evidence-frames" className="border-l-4 border-amber-500 bg-amber-50 p-5 text-amber-950">
        <h2 className="text-lg font-bold">Safety findings detected</h2>
        <p className="mt-1 text-sm">Visual evidence is unavailable for this result. Review the finding summary and original footage.</p>
      </section>
    );
  }

  if (!ownedFrames.length && acceptedFindingCount === 0) {
    return (
      <section id="evidence-frames" className="border-l-4 border-emerald-500 bg-emerald-50 p-5 text-emerald-950">
        <h2 className="text-lg font-bold">No violations found.</h2>
        <p className="mt-1 text-sm">No evidence frames were generated.</p>
      </section>
    );
  }

  if (!ownedFrames.length) {
    return null;
  }

  return (
    <section id="evidence-frames">
      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-950">Evidence Frames</h2>
          <p className="mt-1 text-sm text-slate-600">
            Review the frames that support each safety finding.
          </p>
        </div>
        <a
          className="focus-ring inline-flex w-fit items-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-safety-blue hover:text-safety-blue"
          href="#video-violation-overview"
        >
          Back to summary
        </a>
      </div>

      <div className="mb-4 flex flex-wrap items-end gap-3 border-y border-slate-200 bg-white py-3">
        <label className="text-xs font-bold uppercase text-slate-600">Finding
          <select value={findingFilter} onChange={(event) => setFindingFilter(event.target.value)} className="ml-2 rounded border border-slate-300 bg-white px-2 py-1.5 text-sm font-semibold normal-case text-slate-800">
            <option value="all">All findings</option>
            {findings.map((finding) => <option key={finding} value={finding}>{finding}</option>)}
          </select>
        </label>
        <label className="text-xs font-bold uppercase text-slate-600">Sort
          <select value={sortMode} onChange={(event) => setSortMode(event.target.value as 'timestamp' | 'confidence')} className="ml-2 rounded border border-slate-300 bg-white px-2 py-1.5 text-sm font-semibold normal-case text-slate-800">
            <option value="timestamp">Timestamp</option><option value="confidence">Review confidence</option>
          </select>
        </label>
        <div className="flex rounded border border-slate-300 bg-white">
          <button type="button" title="List view" onClick={() => setViewMode('list')} className={`focus-ring p-2 ${viewMode === 'list' ? 'text-safety-blue' : 'text-slate-500'}`}><List className="h-4 w-4" /></button>
          <button type="button" title="Gallery view" onClick={() => setViewMode('gallery')} className={`focus-ring p-2 ${viewMode === 'gallery' ? 'text-safety-blue' : 'text-slate-500'}`}><Grid2X2 className="h-4 w-4" /></button>
        </div>
        <button type="button" onClick={exportEvidenceMetadata} className="focus-ring inline-flex items-center gap-2 rounded border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700"><Download className="h-4 w-4" />Export filtered metadata</button>
      </div>

      {foreignFrameCount > 0 ? (
        <div role="alert" className="mb-4 flex items-start gap-2 rounded border border-red-300 bg-red-50 p-3 text-sm font-semibold text-red-800">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          SafeTrace blocked {foreignFrameCount} evidence item{foreignFrameCount === 1 ? '' : 's'} whose job ownership did not match the selected result.
        </div>
      ) : null}

      <div className={`grid gap-4 ${viewMode === 'gallery' ? 'xl:grid-cols-2' : ''}`}>
        {visibleFrames.map((frame) => (
          <FrameEvidenceCard
            key={`${jobId || 'preview'}:${frame.evidenceId || frame.id}`}
            frame={frame}
            showExplanation={showExplanations}
            isHighlighted={frame.id === highlightedFrameId}
            jobId={jobId}
            useCaseProfile={useCaseProfile}
            effectiveQuery={effectiveQuery}
            analysisDiagnostics={analysisDiagnostics}
            presentationNumber={presentationNumbers.get(frame.evidenceId || frame.id) ?? 1}
            totalEvidence={canonicalFrames.length}
          />
        ))}
      </div>
    </section>
  );
}
