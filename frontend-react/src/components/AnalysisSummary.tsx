import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Copy,
  Eye,
  FileVideo,
  Gauge,
  Layers3,
  ListChecks,
  ShieldAlert,
  type LucideIcon,
} from 'lucide-react';
import type { AnalysisResult, Severity } from '../types/analysis';
import { isViolationAlignedWithProfile, supportLevelLabel } from '../data/useCaseProfiles';
import { formatConfidence, formatDateTime, pluralize } from '../utils/formatters';
import { copyJobIdToClipboard, formatShortJobId } from '../utils/jobIds';
import { SeverityBadge } from './SeverityBadge';
import { StatusBadge } from './StatusBadge';

type AnalysisSummaryProps = {
  result: AnalysisResult;
  showExplanations: boolean;
};

type KeyFinding = {
  id: string;
  name: string;
  severity: Severity;
  eventCount: number;
  firstTimestamp: string;
  lastTimestamp: string;
  confidence: number;
  supportingFrameCount: number;
  firstFrameId?: string;
};

type SummaryModel = {
  decision: 'Violations detected' | 'No violations detected' | 'Needs review';
  decisionTone: 'danger' | 'success' | 'warning';
  confidence: number | null;
  confidenceExplanation: string;
  framesWithViolations: number;
  framesWithoutViolations: number;
  violationTypeCount: number;
  eventCount: number;
  supportingFrameCount: number;
  highestSeverity?: Severity;
  strongestFinding?: KeyFinding;
  keyFindings: KeyFinding[];
  hiddenProfileFindingCount: number;
  nextAction: string;
  samplingFps?: number;
  topK?: number;
  poolingStrategy?: string;
  batchId?: string;
};

const severityRank: Record<Severity, number> = {
  High: 3,
  Medium: 2,
  Low: 1,
};

function toSeverity(value: string | undefined | null): Severity {
  const normalized = (value || '').toLowerCase();
  if (normalized === 'high' || normalized === 'critical') return 'High';
  if (normalized === 'medium') return 'Medium';
  return 'Low';
}

function timestampValue(timestamp: string): number {
  const parts = timestamp.split(':').map(Number);
  if (parts.length !== 3 || parts.some((part) => !Number.isFinite(part))) return 0;
  return parts[0] * 3600 + parts[1] * 60 + parts[2];
}

function earlierTimestamp(current: string, next: string): string {
  return timestampValue(next) < timestampValue(current) ? next : current;
}

function laterTimestamp(current: string, next: string): string {
  return timestampValue(next) > timestampValue(current) ? next : current;
}

function getTechnicalValue(result: AnalysisResult, key: string): unknown {
  const metadata = result.technicalDetails?.processingMetadata;
  if (typeof metadata === 'object' && metadata !== null && key in metadata) {
    return (metadata as Record<string, unknown>)[key];
  }
  const jobMetrics = result.technicalDetails?.jobMetrics;
  if (typeof jobMetrics === 'object' && jobMetrics !== null && key in jobMetrics) {
    return (jobMetrics as Record<string, unknown>)[key];
  }
  return undefined;
}

function getBatchId(result: AnalysisResult): string | undefined {
  const value = getTechnicalValue(result, 'batchId');
  return typeof value === 'string' ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function formatDurationSeconds(value: unknown): string | null {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return null;
  const totalSeconds = Math.floor(numeric);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function buildKeyFindings(result: AnalysisResult): KeyFinding[] {
  const findings = new Map<string, KeyFinding>();

  function upsert(input: KeyFinding) {
    const existing = findings.get(input.id);
    if (!existing) {
      findings.set(input.id, input);
      return;
    }
    existing.eventCount += input.eventCount;
    existing.firstTimestamp = earlierTimestamp(existing.firstTimestamp, input.firstTimestamp);
    existing.lastTimestamp = laterTimestamp(existing.lastTimestamp, input.lastTimestamp);
    existing.confidence = Math.max(existing.confidence, input.confidence);
    existing.supportingFrameCount += input.supportingFrameCount;
    if (severityRank[input.severity] > severityRank[existing.severity]) {
      existing.severity = input.severity;
    }
    existing.firstFrameId = existing.firstFrameId || input.firstFrameId;
  }

  if (result.events?.length) {
    result.events.forEach((event) => {
      upsert({
        id: event.type,
        name: event.name,
        severity: event.severity,
        eventCount: 1,
        firstTimestamp: event.startTimestamp,
        lastTimestamp: event.endTimestamp,
        confidence: event.representativeConfidence,
        supportingFrameCount: event.supportingFrameCount,
        firstFrameId: event.supportingFrames[0]?.frameId,
      });
    });
  } else if (result.violations?.length) {
    result.violations.forEach((violation) => {
      const timestamps = violation.affectedFrames.map((frame) => frame.timestamp);
      upsert({
        id: violation.id,
        name: violation.name,
        severity: toSeverity(violation.severity),
        eventCount: 1,
        firstTimestamp: timestamps.reduce(earlierTimestamp, timestamps[0] || '00:00:00'),
        lastTimestamp: timestamps.reduce(laterTimestamp, timestamps[0] || '00:00:00'),
        confidence: violation.confidenceMax,
        supportingFrameCount: violation.affectedFrames.length,
        firstFrameId: violation.affectedFrames[0]?.frameId,
      });
    });
  } else {
    result.frames.forEach((frame) => {
      frame.violations.forEach((violation) => {
        upsert({
          id: violation.type,
          name: violation.name,
          severity: violation.severity,
          eventCount: 1,
          firstTimestamp: frame.timestamp,
          lastTimestamp: frame.timestamp,
          confidence: violation.confidence,
          supportingFrameCount: 1,
          firstFrameId: frame.id,
        });
      });
    });
  }

  return Array.from(findings.values()).sort((a, b) => {
    const severityDelta = severityRank[b.severity] - severityRank[a.severity];
    if (severityDelta !== 0) return severityDelta;
    return b.confidence - a.confidence;
  });
}

function buildSummaryModel(result: AnalysisResult): SummaryModel {
  const useCaseProfile = result.settings?.useCaseProfile ?? result.media.useCaseProfile;
  const allKeyFindings = buildKeyFindings(result);
  const keyFindings = useCaseProfile
    ? allKeyFindings.filter((finding) => isViolationAlignedWithProfile(useCaseProfile, finding.name || finding.id))
    : allKeyFindings;
  const hiddenProfileFindingCount = allKeyFindings.length - keyFindings.length;
  const framesWithViolations = result.summary?.framesWithViolations
    ?? result.frames.filter((frame) => frame.violations.length > 0).length;
  const framesWithoutViolations = Math.max(result.framesAnalyzed - framesWithViolations, 0);
  const eventCount = result.summary?.potentialEventCount ?? result.events?.length ?? keyFindings.length;
  const confidence = typeof result.summary?.overallConfidence === 'number'
    ? result.summary.overallConfidence
    : keyFindings[0]?.confidence ?? null;
  const highestSeverity = keyFindings[0]?.severity;
  const strongestFinding = [...keyFindings].sort((a, b) => b.confidence - a.confidence)[0];
  const supportingFrameCount = keyFindings.reduce((sum, finding) => sum + finding.supportingFrameCount, 0);
  const decision = keyFindings.length
    ? 'Violations detected'
    : result.framesAnalyzed > 0
      ? 'No violations detected'
      : 'Needs review';
  const decisionTone = decision === 'Violations detected'
    ? 'danger'
    : decision === 'No violations detected'
      ? 'success'
      : 'warning';
  const confidenceExplanation = confidence === null
    ? 'No aggregate confidence was available; review frame-level evidence before deciding.'
    : confidence >= 0.85
      ? 'High-confidence evidence is present, but the reviewer should still confirm the frames.'
      : confidence >= 0.6
        ? 'Moderate confidence; inspect supporting frames before taking action.'
        : 'Low confidence or limited support; treat this as a review cue rather than a decision.';
  const nextAction = keyFindings[0]?.firstFrameId
    ? `Start with ${keyFindings[0].name} at ${keyFindings[0].firstTimestamp}.`
    : 'Review the sampled frames and confirm whether the query matches the scene.';

  return {
    decision,
    decisionTone,
    confidence,
    confidenceExplanation,
    framesWithViolations,
    framesWithoutViolations,
    violationTypeCount: result.summary?.uniqueViolationTypes ?? keyFindings.length,
    eventCount,
    supportingFrameCount,
    highestSeverity,
    strongestFinding,
    keyFindings,
    hiddenProfileFindingCount,
    nextAction,
    samplingFps: result.settings?.fps ?? numberValue(getTechnicalValue(result, 'fps')),
    topK: result.settings?.topK,
    poolingStrategy: String(getTechnicalValue(result, 'embeddingPoolingStrategy') || ''),
    batchId: getBatchId(result),
  };
}

export function AnalysisSummary({ result, showExplanations }: AnalysisSummaryProps) {
  const summary = buildSummaryModel(result);
  const hasViolations = summary.keyFindings.length > 0;
  const jobId = result.jobId;
  const shortJobId = formatShortJobId(jobId);
  const useCaseProfile = result.settings?.useCaseProfile ?? result.media.useCaseProfile;
  const completedDuration = formatDurationSeconds(result.elapsedSeconds ?? getTechnicalValue(result, 'elapsedSeconds'));
  const analysisRuntime = formatDurationSeconds(result.analysisRuntimeSeconds ?? getTechnicalValue(result, 'analysisRuntimeSeconds'));
  const evidenceHref = summary.strongestFinding?.firstFrameId
    ? `#frame-${summary.strongestFinding.firstFrameId}`
    : '#evidence-frames';

  return (
    <section id="analysis-summary" className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold text-safety-blue">Analysis summary</p>
          <h2 className="mt-1 text-2xl font-bold tracking-normal text-slate-950">
            {summary.decision}
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            SafeTrace analyzed <span className="font-semibold text-slate-800">{result.media.filename}</span>
            {summary.batchId ? ` from batch ${summary.batchId}` : ''} for
            {' '}<span className="font-semibold text-slate-800">{result.query}</span>.{' '}
            {hasViolations
              ? `${summary.violationTypeCount} violation type${summary.violationTypeCount === 1 ? '' : 's'} and ${summary.eventCount} grouped event${summary.eventCount === 1 ? '' : 's'} need review.`
              : 'No matching violation findings were detected in the selected evidence frames.'}
          </p>
          {showExplanations && result.summaryText ? (
            <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm leading-6 text-blue-900">
              Explanation generated from available visual evidence. {result.summaryText}
            </div>
          ) : null}
          {useCaseProfile ? (
            <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700">
              <span className="font-semibold uppercase text-slate-500">Use-case profile</span>
              <span className="ml-2 font-semibold text-slate-900">{useCaseProfile.label}</span>
              <span className="ml-2 text-slate-500">{useCaseProfile.category}</span>
              <span className="ml-2 rounded-full border border-slate-200 bg-white px-2 py-0.5 font-semibold uppercase text-slate-500">
                {supportLevelLabel(useCaseProfile.backendSupportLevel)}
              </span>
              <p className="mt-1">{useCaseProfile.description}</p>
              <p className="mt-1">
                Effective query used: <span className="font-semibold text-slate-900">{useCaseProfile.effectiveQuery ?? result.query}</span>
              </p>
              {useCaseProfile.supportedChecks?.length ? (
                <p className="mt-1">Intended checks: {useCaseProfile.supportedChecks.join(', ')}</p>
              ) : null}
              {useCaseProfile.limitations ? (
                <p className="mt-1 text-amber-800">{useCaseProfile.limitations}</p>
              ) : null}
              {useCaseProfile.customText ? (
                <p className="mt-1 font-medium text-slate-800">Policy note: {useCaseProfile.customText}</p>
              ) : null}
            </div>
          ) : null}
          {jobId ? (
            <div className="mt-4 flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              <span className="font-semibold uppercase text-slate-500">Result job</span>
              <code className="rounded bg-white px-2 py-1 font-mono text-[11px] font-semibold text-slate-900" title={jobId}>
                {shortJobId}
              </code>
              <button
                type="button"
                onClick={() => void copyJobIdToClipboard(jobId)}
                className="focus-ring inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 font-semibold text-slate-700 transition hover:border-safety-blue hover:text-safety-blue"
                aria-label="Copy job ID"
                title={`Copy full job ID ${jobId}`}
              >
                <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                Copy job ID
              </button>
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          {summary.highestSeverity ? <SeverityBadge severity={summary.highestSeverity} /> : null}
          <StatusBadge label={summary.decision} tone={summary.decisionTone} />
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <SummaryMetric icon={FileVideo} label="Media analyzed" value={result.media.filename} />
        {useCaseProfile ? <SummaryMetric icon={ListChecks} label="Use-case profile" value={useCaseProfile.label} /> : null}
        {jobId ? <SummaryMetric icon={FileVideo} label="Job ID" value={shortJobId} /> : null}
        {completedDuration ? <SummaryMetric icon={Clock3} label="Completed in" value={completedDuration} /> : null}
        <SummaryMetric icon={ShieldAlert} label="Review confidence" value={summary.confidence === null ? 'Needs review' : formatConfidence(summary.confidence)} />
        <SummaryMetric icon={Layers3} label="Grouped events" value={String(summary.eventCount)} />
        <SummaryMetric icon={Clock3} label="Generated" value={formatDateTime(result.generatedAt)} />
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-[1.3fr_0.9fr]">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-950">
            <ListChecks className="h-4 w-4 text-safety-blue" aria-hidden="true" />
            Key findings
          </div>
          {summary.keyFindings.length ? (
            <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
              <div className="grid gap-3 bg-slate-50 px-3 py-2 text-xs font-bold uppercase text-slate-500 md:grid-cols-[1.2fr_0.7fr_0.8fr_0.8fr_0.8fr_0.8fr]">
                <span>Type</span>
                <span>Events</span>
                <span>First</span>
                <span>Last</span>
                <span>Evidence strength</span>
                <span>Frames</span>
              </div>
              {summary.keyFindings.map((finding) => (
                <div
                  key={finding.id}
                  className="grid gap-3 border-t border-slate-200 px-3 py-2 text-sm md:grid-cols-[1.2fr_0.7fr_0.8fr_0.8fr_0.8fr_0.8fr] md:items-center"
                >
                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                    <span className="truncate font-semibold text-slate-950">{finding.name}</span>
                    <SeverityBadge severity={finding.severity} />
                  </div>
                  <span>{finding.eventCount}</span>
                  <span className="font-mono text-xs">{finding.firstTimestamp}</span>
                  <span className="font-mono text-xs">{finding.lastTimestamp}</span>
                  <span className="font-semibold">{formatConfidence(finding.confidence)}</span>
                  <span>{finding.supportingFrameCount}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-start gap-2 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              No profile-aligned safety violations were detected in the selected frames.
            </div>
          )}
          {summary.hiddenProfileFindingCount ? (
            <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
              {summary.hiddenProfileFindingCount} raw backend finding{summary.hiddenProfileFindingCount === 1 ? '' : 's'} were outside the selected profile and are de-prioritized from Key findings. Review technical evidence before treating any candidate as an operational incident.
            </p>
          ) : null}
        </div>

        <div className="grid gap-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-950">
              <Gauge className="h-4 w-4 text-safety-blue" aria-hidden="true" />
              Evidence coverage
            </div>
            <dl className="grid gap-2 text-sm text-slate-700">
              <SummaryLine label="Frames sampled" value={String(result.framesAnalyzed)} />
              <SummaryLine label="Frames with findings" value={String(summary.framesWithViolations)} />
              <SummaryLine label="Frames without findings" value={String(summary.framesWithoutViolations)} />
              <SummaryLine label="Sampling FPS" value={summary.samplingFps ? summary.samplingFps.toFixed(1) : 'Not reported'} />
              <SummaryLine label="Top-K frame limit" value={summary.topK ? String(summary.topK) : 'Not reported'} />
              <SummaryLine label="Pooling strategy" value={summary.poolingStrategy || 'Not reported'} />
              <SummaryLine label="Analysis runtime" value={analysisRuntime || 'Not reported'} />
            </dl>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-bold text-slate-950">
              <Eye className="h-4 w-4 text-safety-blue" aria-hidden="true" />
              Review guidance
            </div>
            <p className="text-sm leading-6 text-slate-600">{summary.nextAction}</p>
            <p className="mt-2 text-sm leading-6 text-slate-600">{summary.confidenceExplanation}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <a
                className="focus-ring inline-flex items-center rounded-lg bg-safety-blue px-3 py-2 text-xs font-semibold text-white transition hover:bg-blue-700"
                href={evidenceHref}
              >
                Review supporting evidence
              </a>
              <a
                className="focus-ring inline-flex items-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-safety-blue hover:text-safety-blue"
                href="#video-violation-overview"
              >
                Open violation overview
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-950">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" aria-hidden="true" />
          <p>
            Limitations: SafeTrace findings are automated review aids, not final operational or legal truth.
            Confirm important findings against the original footage, video quality, sampled frames, and technical evidence.
          </p>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <StatusBadge label={`Query: ${result.query}`} tone="info" />
        {useCaseProfile ? <StatusBadge label={`Profile: ${useCaseProfile.label}`} tone="info" /> : null}
        {useCaseProfile ? <StatusBadge label={supportLevelLabel(useCaseProfile.backendSupportLevel)} tone="warning" /> : null}
        {jobId ? <StatusBadge label={`Job ${shortJobId}`} tone="neutral" /> : null}
        <StatusBadge label={pluralize(summary.violationTypeCount, 'violation type')} tone={hasViolations ? 'danger' : 'success'} />
        <StatusBadge label={`${summary.supportingFrameCount} supporting frame${summary.supportingFrameCount === 1 ? '' : 's'}`} tone="neutral" />
        {summary.batchId ? <StatusBadge label={`Batch ${summary.batchId}`} tone="neutral" /> : null}
      </div>
    </section>
  );
}

type SummaryMetricProps = {
  icon: LucideIcon;
  label: string;
  value: string;
};

function SummaryMetric({ icon: Icon, label, value }: SummaryMetricProps) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-white text-safety-blue shadow-insetLine">
        <Icon className="h-4 w-4" aria-hidden="true" />
      </div>
      <p className="text-xs font-semibold uppercase text-slate-500">{label}</p>
      <p className="mt-1 break-words text-sm font-semibold leading-5 text-slate-950">{value}</p>
    </div>
  );
}

function SummaryLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-slate-200 pb-2 last:border-b-0 last:pb-0">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-semibold text-slate-950">{value}</dd>
    </div>
  );
}
