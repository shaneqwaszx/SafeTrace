import { CheckCircle2, LoaderCircle } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

type AnalysisProgressProps = {
  steps: string[];
  activeStep: number;
  currentStep?: string;
  progress?: number;
  progressPercent?: number;
  stage?: string;
  message?: string;
  mode?: 'backend' | 'preview' | null;
  status?: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string | null;
  createdAt?: string | null;
  queuedAt?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  failedAt?: string | null;
  cancelledAt?: string | null;
  elapsedSeconds?: number | null;
  queueWaitSeconds?: number | null;
  analysisRuntimeSeconds?: number | null;
  updatedAt?: string | null;
  heartbeatAt?: string | null;
};

function formatDurationSeconds(secondsValue: number): string {
  const totalSeconds = Math.max(0, Math.floor(secondsValue));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function formatElapsed(ms: number): string {
  return formatDurationSeconds(ms / 1000);
}

function elapsedStatusLabel(status: AnalysisProgressProps['status'] | undefined | null) {
  if (status === 'queued') return 'Queued for';
  if (status === 'completed') return 'Completed in';
  if (status === 'failed') return 'Failed after';
  if (status === 'cancelled') return 'Cancelled after';
  return 'Running for';
}

function elapsedFromServerSeconds({
  status,
  elapsedSeconds,
  createdAt,
  queuedAt,
  startedAt,
  completedAt,
  failedAt,
  cancelledAt,
  now,
}: {
  status?: AnalysisProgressProps['status'];
  elapsedSeconds?: number | null;
  createdAt?: string | null;
  queuedAt?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  failedAt?: string | null;
  cancelledAt?: string | null;
  now: number;
}): number | null {
  const terminal = status === 'completed' || status === 'failed' || status === 'cancelled';
  if (terminal && typeof elapsedSeconds === 'number') return elapsedSeconds;
  const start = parseTime(createdAt) ?? parseTime(queuedAt) ?? parseTime(startedAt);
  if (!start) return typeof elapsedSeconds === 'number' ? elapsedSeconds : null;
  const end = status === 'completed'
    ? parseTime(completedAt)
    : status === 'failed'
      ? parseTime(failedAt)
      : status === 'cancelled'
        ? parseTime(cancelledAt)
        : now;
  if (!end) return typeof elapsedSeconds === 'number' ? elapsedSeconds : null;
  return Math.max(0, (end - start) / 1000);
}

function parseTime(value?: string | null): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function heartbeatLabel(now: number, value?: string | null): string {
  const parsed = parseTime(value);
  if (!parsed) return 'Waiting for backend heartbeat';
  const seconds = Math.max(0, Math.floor((now - parsed) / 1000));
  if (seconds < 3) return 'Backend heartbeat just now';
  if (seconds < 60) return `Backend heartbeat ${seconds}s ago`;
  return `Backend heartbeat ${Math.floor(seconds / 60)}m ago`;
}

export function AnalysisProgress({
  steps,
  activeStep,
  currentStep,
  progress,
  progressPercent,
  stage,
  message,
  mode,
  status,
  createdAt,
  queuedAt,
  startedAt,
  completedAt,
  failedAt,
  cancelledAt,
  elapsedSeconds,
  queueWaitSeconds,
  analysisRuntimeSeconds,
  updatedAt,
  heartbeatAt,
}: AnalysisProgressProps) {
  const [now, setNow] = useState(() => Date.now());
  const [localStartedAt] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const computedPercent = typeof progressPercent === 'number'
    ? progressPercent
    : typeof progress === 'number'
      ? Math.round(progress * 100)
      : null;
  const isIndeterminateAnalysis = mode === 'backend' && stage === 'analyzing' && (computedPercent ?? 0) < 85;
  const statusMessage = message || currentStep || 'Submitting media to the local SafeTrace backend.';
  const startedMs = useMemo(() => parseTime(startedAt) ?? localStartedAt, [startedAt, localStartedAt]);
  const serverElapsedSeconds = elapsedFromServerSeconds({
    status,
    elapsedSeconds,
    createdAt,
    queuedAt,
    startedAt,
    completedAt,
    failedAt,
    cancelledAt,
    now,
  });
  const elapsed = mode === 'backend'
    ? formatDurationSeconds(serverElapsedSeconds ?? ((now - startedMs) / 1000))
    : null;
  const elapsedLabel = elapsedStatusLabel(status);
  const heartbeat = mode === 'backend' ? heartbeatLabel(now, heartbeatAt || updatedAt) : null;
  const queueRuntimeCopy = queueWaitSeconds || analysisRuntimeSeconds
    ? [
      typeof queueWaitSeconds === 'number' ? `Queue ${formatDurationSeconds(queueWaitSeconds)}` : null,
      typeof analysisRuntimeSeconds === 'number' ? `Analysis ${formatDurationSeconds(analysisRuntimeSeconds)}` : null,
    ].filter(Boolean).join(' | ')
    : null;

  return (
    <section className="rounded-lg border border-blue-200 bg-blue-50 p-5 text-blue-950 shadow-soft">
      <div className="flex items-start gap-3">
        <LoaderCircle className="mt-0.5 h-5 w-5 shrink-0 animate-spin" aria-hidden="true" />
        <div>
          <h2 className="text-sm font-bold">
            {mode === 'preview' ? 'Preparing developer preview...' : 'Analyzing selected footage...'}
          </h2>
          <p className="mt-1 text-sm leading-6">
            {statusMessage}
            {!isIndeterminateAnalysis && computedPercent !== null ? ` ${computedPercent}%` : ''}
          </p>
          {stage ? <p className="mt-1 text-xs font-semibold uppercase text-blue-700">Stage: {stage}</p> : null}
          {mode === 'backend' ? (
            <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold text-blue-800">
              <span className="rounded-full border border-blue-200 bg-white/70 px-2.5 py-1">{elapsedLabel}: {elapsed}</span>
              {queueRuntimeCopy ? (
                <span className="rounded-full border border-blue-200 bg-white/70 px-2.5 py-1">{queueRuntimeCopy}</span>
              ) : null}
              <span className="rounded-full border border-blue-200 bg-white/70 px-2.5 py-1">{heartbeat}</span>
              {isIndeterminateAnalysis ? (
                <span className="rounded-full border border-blue-200 bg-white/70 px-2.5 py-1">
                  Still working locally; long videos can spend time in sampling, embedding, detection, and report prep.
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-full bg-white/80">
        {isIndeterminateAnalysis ? (
          <div className="h-2 w-1/2 animate-pulse rounded-full bg-blue-500" />
        ) : (
          <div
            className="h-2 rounded-full bg-blue-600 transition-all duration-500"
            style={{ width: `${Math.max(4, Math.min(100, computedPercent ?? 10))}%` }}
          />
        )}
      </div>

      <ol className="mt-5 grid gap-3 md:grid-cols-5">
        {steps.map((step, index) => {
          const isComplete = index < activeStep;
          const isActive = index === activeStep;

          return (
            <li
              key={step}
              className={`rounded-lg border p-3 text-sm ${
                isComplete
                  ? 'border-emerald-200 bg-white text-emerald-800'
                  : isActive
                    ? 'border-blue-300 bg-white text-blue-900'
                    : 'border-blue-100 bg-blue-100/60 text-blue-700'
              }`}
            >
              <div className="mb-2 flex items-center gap-2">
                {isComplete ? (
                  <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${isActive ? 'animate-pulse bg-blue-600' : 'bg-blue-300'}`}
                  />
                )}
                <span className="text-xs font-semibold uppercase">Step {index + 1}</span>
              </div>
              <p className="font-semibold leading-5">{step}</p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
