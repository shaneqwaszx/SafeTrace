import type { FrameResult, UseCaseProfileSelection } from '../types/analysis';
import { FrameEvidenceCard } from './FrameEvidenceCard';

type EvidenceFramesProps = {
  frames: FrameResult[];
  showExplanations: boolean;
  highlightedFrameId?: string | null;
  jobId?: string | null;
  useCaseProfile?: UseCaseProfileSelection;
  effectiveQuery?: string;
  analysisDiagnostics?: Record<string, unknown> | null;
};

export function EvidenceFrames({
  frames,
  showExplanations,
  highlightedFrameId,
  jobId,
  useCaseProfile,
  effectiveQuery,
  analysisDiagnostics,
}: EvidenceFramesProps) {
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

      <div className="grid gap-4">
        {frames.map((frame) => (
          <FrameEvidenceCard
            key={frame.id}
            frame={frame}
            showExplanation={showExplanations}
            isHighlighted={frame.id === highlightedFrameId}
            jobId={jobId}
            useCaseProfile={useCaseProfile}
            effectiveQuery={effectiveQuery}
            analysisDiagnostics={analysisDiagnostics}
          />
        ))}
      </div>
    </section>
  );
}
