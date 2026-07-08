import { AlertTriangle, Frame } from 'lucide-react';
import type { Severity } from '../types/analysis';
import { formatAverageConfidence, formatConfidence, formatConfidenceRange } from '../utils/formatters';
import { SeverityBadge } from './SeverityBadge';

export type GroupedViolation = {
  type: string;
  name: string;
  severity: Severity;
  description: string;
  affectedFrames: Array<{
    frameId: string;
    frameNumber: number;
    timestamp: string;
  }>;
  confidences: number[];
  startTimestamp?: string;
  endTimestamp?: string;
  representativeConfidence?: number;
  supportingFrameCount?: number;
};

type ViolationCardProps = {
  violation: GroupedViolation;
  onFrameSelect: (frameId: string) => void;
};

export function ViolationCard({ violation, onFrameSelect }: ViolationCardProps) {
  const firstFrameId = violation.affectedFrames[0]?.frameId;
  const isAggregatedEvent = Boolean(violation.startTimestamp && violation.endTimestamp);
  const visibleFrames = violation.affectedFrames.slice(0, 6);
  const hiddenFrameCount = Math.max(violation.affectedFrames.length - visibleFrames.length, 0);

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft transition hover:border-slate-300">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-600" aria-hidden="true" />
            <h3 className="text-base font-bold text-slate-950">{violation.name}</h3>
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">{violation.description}</p>
        </div>
        <SeverityBadge severity={violation.severity} />
      </div>

      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        <div className="rounded-lg bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase text-slate-500">
            {isAggregatedEvent ? 'Event evidence' : 'Affected evidence'}
          </p>
          {isAggregatedEvent ? (
            <p className="mt-2 text-sm font-semibold text-slate-950">
              {violation.startTimestamp} - {violation.endTimestamp}
            </p>
          ) : null}
          <div className="mt-2 flex flex-wrap gap-2">
            {visibleFrames.map((frame) => (
              <button
                key={`${violation.type}-${frame.frameNumber}`}
                className="focus-ring inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:border-safety-blue hover:text-safety-blue"
                type="button"
                onClick={() => onFrameSelect(frame.frameId)}
              >
                <Frame className="h-3.5 w-3.5" aria-hidden="true" />
                Frame {frame.frameNumber} at {frame.timestamp}
              </button>
            ))}
            {hiddenFrameCount > 0 ? (
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
                +{hiddenFrameCount} more
              </span>
            ) : null}
          </div>
        </div>

        <div className="rounded-lg bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase text-slate-500">Evidence strength</p>
          <p className="mt-2 text-sm font-semibold text-slate-950">
            {typeof violation.representativeConfidence === 'number'
              ? formatConfidence(violation.representativeConfidence)
              : formatConfidenceRange(violation.confidences)}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {isAggregatedEvent
              ? `${violation.supportingFrameCount ?? violation.affectedFrames.length} supporting frame${(violation.supportingFrameCount ?? violation.affectedFrames.length) === 1 ? '' : 's'}`
              : `Average ${formatAverageConfidence(violation.confidences)}`}
          </p>
        </div>
      </div>

      {firstFrameId ? (
        <button
          className="focus-ring mt-4 inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-safety-blue hover:text-safety-blue"
          type="button"
          onClick={() => onFrameSelect(firstFrameId)}
        >
          View evidence
        </button>
      ) : null}
    </article>
  );
}
