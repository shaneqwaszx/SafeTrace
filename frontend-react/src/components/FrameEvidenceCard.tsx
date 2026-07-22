import clsx from 'clsx';
import { CheckCircle2, ShieldAlert, Sparkles } from 'lucide-react';
import type { FrameResult, UseCaseProfileSelection, Violation } from '../types/analysis';
import { isViolationAlignedWithProfile, profileFindingContext, supportLevelLabel } from '../data/useCaseProfiles';
import { formatConfidence, formatQueryRelevance } from '../utils/formatters';
import { EvidenceFrameVisual } from './EvidenceFrameVisual';
import { SeverityBadge } from './SeverityBadge';
import { StatusBadge } from './StatusBadge';
import { TechnicalDetails } from './TechnicalDetails';

type FrameEvidenceCardProps = {
  frame: FrameResult;
  showExplanation: boolean;
  isHighlighted?: boolean;
  jobId?: string | null;
  useCaseProfile?: UseCaseProfileSelection;
  effectiveQuery?: string;
  analysisDiagnostics?: Record<string, unknown> | null;
  presentationNumber: number;
  totalEvidence: number;
};

function mobileSamRefinement(frame: FrameResult): Record<string, unknown> | null {
  const searchMetadata = frame.technicalEvidence?.searchMetadata;
  if (!searchMetadata || typeof searchMetadata !== 'object') return null;
  const refinement = (searchMetadata as Record<string, unknown>).mobileSamRefinement;
  return refinement && typeof refinement === 'object' ? refinement as Record<string, unknown> : null;
}

function lightweightVlmExplanation(frame: FrameResult): Record<string, unknown> | null {
  const searchMetadata = frame.technicalEvidence?.searchMetadata;
  if (!searchMetadata || typeof searchMetadata !== 'object') return null;
  const explanation = (searchMetadata as Record<string, unknown>).lightweightVlmExplanation;
  return explanation && typeof explanation === 'object' ? explanation as Record<string, unknown> : null;
}

function metadataString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function evidenceStrengthLabel(value: unknown): string {
  const normalized = typeof value === 'string' ? value : '';
  const labels: Record<string, string> = {
    confirmed_violation: 'Strong visual evidence',
    likely_violation: 'Likely issue',
    review_candidate: 'Review cue',
    insufficient_evidence: 'Insufficient evidence',
    unsupported_rule: 'Not enough model support',
  };
  return labels[normalized] || 'Review cue';
}

function verifierAgreementLabel(value: unknown): string | null {
  if (value === 'supports') return 'Visual review supports the finding';
  if (value === 'inconclusive') return 'Visual review inconclusive';
  if (value === 'disagrees') return 'Visual review may disagree with the base finding';
  return null;
}

function evidenceModeDiagnostics(frame: FrameResult, analysisDiagnostics?: Record<string, unknown> | null) {
  const searchMetadata = frame.technicalEvidence?.searchMetadata;
  const metadata = searchMetadata && typeof searchMetadata === 'object' ? searchMetadata as Record<string, unknown> : {};
  const vlmWorker = metadata.lightweightVlmExplanation && typeof metadata.lightweightVlmExplanation === 'object'
    ? metadata.lightweightVlmExplanation as Record<string, unknown>
    : {};
  return {
    requestedMode: metadataString(metadata.requestedVisualExplanationMode)
      ?? metadataString(analysisDiagnostics?.requestedVisualExplanationMode)
      ?? metadataString(analysisDiagnostics?.vlmProfile),
    actualMode: metadataString(metadata.actualExplanationMode)
      ?? metadataString(analysisDiagnostics?.actualExplanationMode)
      ?? metadataString(analysisDiagnostics?.effectiveExplanationMode)
      ?? (frame.explanationSource || 'rule_based'),
    workerAttempted: Boolean(vlmWorker.lightweightVlmWorkerAttempted ?? analysisDiagnostics?.lightweightVlmWorkerAttempted),
    workerSucceeded: Boolean(vlmWorker.lightweightVlmWorkerSucceeded ?? analysisDiagnostics?.lightweightVlmWorkerSucceeded),
    fallbackReason: metadataString(vlmWorker.lightweightVlmFallbackReason)
      ?? metadataString(analysisDiagnostics?.lightweightVlmFallbackReason)
      ?? metadataString(analysisDiagnostics?.fallbackReason),
    baseSource: metadataString(vlmWorker.baseExplanationSource)
      ?? metadataString(analysisDiagnostics?.baseExplanationSource)
      ?? 'rule_based',
    finalSource: metadataString(vlmWorker.finalExplanationSource)
      ?? metadataString(analysisDiagnostics?.finalExplanationSource),
    lightweightContributionAccepted: Boolean(
      vlmWorker.lightweightVlmContributionAccepted ?? analysisDiagnostics?.lightweightVlmContributionAccepted,
    ),
    enhancedLayerStatus: metadataString(vlmWorker.enhancedVlmLayerStatus)
      ?? metadataString(analysisDiagnostics?.enhancedVlmLayerStatus)
      ?? 'unavailable',
  };
}

function friendlyFindingName(violation?: Pick<Violation, 'name' | 'type'> | null): string {
  const raw = String(violation?.name || violation?.type || '').toLowerCase();
  if (raw.includes('seatbelt')) return 'missing seatbelt';
  if (raw.includes('helmet') || raw.includes('ppe')) return 'missing helmet/PPE';
  if (raw.includes('phone')) return 'phone use / distracted driving';
  if (raw.includes('proximity')) return 'unsafe proximity';
  if (!raw) return 'safety finding';
  return raw
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .replace(/^./, (letter) => letter.toLowerCase());
}

function reviewLevelLabel(value: unknown): string {
  const normalized = typeof value === 'string' ? value : '';
  const labels: Record<string, string> = {
    confirmed_violation: 'Strong visual evidence',
    likely_violation: 'Likely issue',
    review_candidate: 'Review cue',
    insufficient_evidence: 'Insufficient evidence',
    unsupported_rule: 'Not enough model support',
  };
  return labels[normalized] || 'Review cue';
}

function structuredVlmField(text: string | null | undefined, field: string): string | null {
  if (!text) return null;
  const fields = 'visible_evidence|visual_status|short_reason|confidence_hint|refined_visual_summary|uncertainty_reason|reviewer_note';
  const pattern = new RegExp(`\\b${field}\\s*:\\s*(.*?)(?=\\b(?:${fields})\\s*:|$)`, 'i');
  const match = pattern.exec(text);
  if (!match) return null;
  const value = match[1].replace(/\s+/g, ' ').trim().replace(/[.:;\s]+$/, '');
  return value || null;
}

function visualReviewFallbackLabel(reason: string | null | undefined): string {
  const raw = (reason || '').toLowerCase();
  if (!raw) return 'Local visual review did not add a confident result for this frame.';
  if (raw.includes('timeout')) return 'Local visual review timed out; Fast Local Analysis fallback shown.';
  if (raw.includes('generic object') || raw.includes('inventory')) return 'Local VLM output was rejected as unrelated object inventory.';
  if (raw.includes('generic person')) return 'Local VLM output was rejected as unrelated person activity.';
  if (raw.includes('prompt') || raw.includes('option-list')) return 'Local VLM output was rejected because it echoed the prompt.';
  if (raw.includes('missing safety') || raw.includes('missing visible')) return 'Local VLM output did not answer the safety question clearly.';
  if (raw.includes('not selected') || raw.includes('frame_limit') || raw.includes('frame limit')) return 'Local visual review was not run for this lower-priority frame.';
  if (raw.includes('model_missing')) return 'Local visual review model was unavailable.';
  if (raw.includes('disabled')) return 'Local visual review was disabled or not loaded.';
  return `Local visual review fallback: ${reason}`;
}

function visualReviewText(vlmWorker: Record<string, unknown> | null, modeDiagnostics: ReturnType<typeof evidenceModeDiagnostics>) {
  const fallbackReason = metadataString(vlmWorker?.lightweightVlmFallbackReason) ?? modeDiagnostics.fallbackReason;
  const cleanPreview = metadataString(vlmWorker?.lightweightVlmCleanTextPreview)
    ?? metadataString(vlmWorker?.lightweightVlmRawTextPreview);
  const visualStatus = structuredVlmField(cleanPreview, 'visual_status');
  const shortReason = structuredVlmField(cleanPreview, 'short_reason')
    ?? structuredVlmField(cleanPreview, 'refined_visual_summary');
  const visibleEvidence = structuredVlmField(cleanPreview, 'visible_evidence');
  const modelProfile = metadataString(vlmWorker?.lightweightVlmModelProfile);
  const modelLabel = modeDiagnostics.enhancedLayerStatus === 'succeeded'
    ? 'Enhanced VLM'
    : modelProfile?.includes('256')
      ? 'Local VLM'
      : 'Local VLM';

  if (modeDiagnostics.workerSucceeded && (shortReason || visualStatus || visibleEvidence)) {
    return `${modelLabel}: ${[visualStatus, shortReason || visibleEvidence].filter(Boolean).join('. ')}.`;
  }
  if (modeDiagnostics.workerSucceeded) {
    return `${modelLabel}: added local visual review for this frame.`;
  }
  if (modeDiagnostics.workerAttempted && fallbackReason?.toLowerCase().includes('timeout')) {
    return 'Local visual review timed out for this frame; Fast Local Analysis remains available.';
  }
  if (modeDiagnostics.workerAttempted) {
    return `${visualReviewFallbackLabel(fallbackReason)} Fast Local Analysis remains available.`;
  }
  if (fallbackReason === 'visual_review_frame_limit_reached' || fallbackReason === 'local_visual_review_not_selected') {
    return 'Local visual review was not run for this lower-priority frame.';
  }
  return 'Fast Local Analysis is available for this frame. Local visual review details are available in technical evidence when enabled.';
}

function whyFlaggedText(violation?: Violation | null): string {
  const name = String(violation?.name || violation?.type || '').toLowerCase();
  if (name.includes('seatbelt')) {
    return 'SafeTrace could not confirm a visible belt path in this evidence frame.';
  }
  if (name.includes('helmet') || name.includes('ppe')) {
    return 'SafeTrace could not confirm visible helmet/PPE evidence in the selected area.';
  }
  if (name.includes('phone')) {
    return 'SafeTrace flagged possible phone or distracted-driving evidence for review.';
  }
  return violation?.description || 'SafeTrace flagged this frame for safety review based on the selected profile and detector evidence.';
}

function whatToCheckText(violation?: Violation | null): string {
  const name = String(violation?.name || violation?.type || '').toLowerCase();
  if (name.includes('seatbelt')) {
    return 'Confirm whether a belt crosses the torso, especially if glare, blur, occlusion, or camera angle affects visibility.';
  }
  if (name.includes('helmet') || name.includes('ppe')) {
    return 'Confirm whether the head and required PPE are visible in the original footage.';
  }
  if (name.includes('phone')) {
    return 'Confirm whether a phone is visible near the hand, face, or driver area and whether active use is clear.';
  }
  return 'Review the original footage before treating this frame as a final operational finding.';
}

function explanationSourceInfo(
  frame: FrameResult,
  vlmWorker: Record<string, unknown> | null,
  modeDiagnostics: ReturnType<typeof evidenceModeDiagnostics>,
) {
  const source = [
    modeDiagnostics.finalSource,
    modeDiagnostics.actualMode,
    frame.explanationSource,
    metadataString(vlmWorker?.lightweightVlmExplanationSource),
  ].filter(Boolean).join(' ').toLowerCase();
  if (source.includes('enhanced')) {
    return {
      header: 'Advanced GPU VLM Assist explanation',
      note: 'SafeTrace used the enhanced GPU visual review to refine the detector/rule evidence.',
    };
  }
  if (source.includes('lightweight') || source.includes('vlm')) {
    return {
      header: 'Local VLM Assist explanation',
      note: 'SafeTrace used local visual review to refine the detector/rule evidence.',
    };
  }
  const requested = [
    modeDiagnostics.requestedMode,
    metadataString(vlmWorker?.lightweightVlmModelProfile),
  ].filter(Boolean).join(' ').toLowerCase();
  const wasVlmRequested = requested.includes('lightweight') || requested.includes('enhanced') || modeDiagnostics.workerAttempted;
  const fallbackNote = visualReviewFallbackLabel(modeDiagnostics.fallbackReason);
  if (modeDiagnostics.workerAttempted && requested.includes('enhanced')) {
    return {
      header: 'Advanced GPU VLM Assist attempted - Fast Local Analysis fallback',
      note: fallbackNote,
    };
  }
  if (modeDiagnostics.workerAttempted && (requested.includes('lightweight') || requested.includes('vlm'))) {
    return {
      header: 'Local VLM Assist attempted - Fast Local Analysis fallback',
      note: fallbackNote,
    };
  }
  return {
    header: 'Fast Local Analysis explanation',
    note: wasVlmRequested
      ? fallbackNote
      : 'SafeTrace used local detector/rule evidence for this frame.',
  };
}

export function FrameEvidenceCard({
  frame,
  showExplanation,
  isHighlighted = false,
  jobId,
  useCaseProfile,
  effectiveQuery,
  analysisDiagnostics,
  presentationNumber,
  totalEvidence,
}: FrameEvidenceCardProps) {
  const displayedViolations = useCaseProfile
    ? frame.violations.filter((violation) => isViolationAlignedWithProfile(useCaseProfile, violation.name || violation.type))
    : frame.violations;
  const hiddenProfileViolationCount = frame.violations.length - displayedViolations.length;
  const hasViolations = displayedViolations.length > 0;
  const refinement = mobileSamRefinement(frame);
  const vlmWorker = lightweightVlmExplanation(frame);
  const modeDiagnostics = evidenceModeDiagnostics(frame, analysisDiagnostics);
  const detectorBoxFallbackUsed = refinement?.mobileSamRefinementSource === 'fallback';
  const primaryViolation = displayedViolations[0] ?? null;
  const visualReviewCopy = visualReviewText(vlmWorker, modeDiagnostics);
  const explanationInfo = explanationSourceInfo(frame, vlmWorker, modeDiagnostics);
  const primaryAgreement = primaryViolation?.verifierAgreement;
  const reviewLevelCopy = primaryAgreement === 'disagrees'
    ? 'Visual review may disagree with the base finding. Do not treat this as an automatic violation or automatic clearance.'
    : primaryAgreement === 'inconclusive'
      ? 'Visual review was inconclusive. Confirm this in the original footage before acting.'
      : `${reviewLevelLabel(primaryViolation?.evidenceStrength)}. Review confidence ${formatConfidence(primaryViolation?.confidence ?? 0)}.`;
  const detectorSummary = frame.detections.length
    ? frame.detections
      .slice(0, 5)
      .map((detection) => `${detection.label} ${Math.round(detection.confidence * 100)}%`)
      .join(', ')
    : 'No detector classes reported';

  return (
    <article
      id={`frame-${frame.id}`}
      className={clsx(
        'scroll-mt-6 overflow-hidden rounded-lg border bg-white shadow-soft transition',
        isHighlighted ? 'border-safety-blue ring-4 ring-blue-100' : 'border-slate-200',
      )}
    >
      <div className="border-b border-slate-200 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="text-base font-bold text-slate-950">
              Frame {presentationNumber} of {totalEvidence}
            </h3>
            <p className="mt-1 text-xs font-semibold text-slate-600">
              Source frame: {frame.sourceFrameIndex ?? 'not reported'} | Timestamp: {frame.timestampLabel || frame.timestamp}
            </p>
            {frame.sourceRelativePath || frame.videoFilename ? (
              <p className="mt-1 text-xs text-slate-500">{frame.sourceRelativePath || frame.videoFilename}</p>
            ) : null}
            {frame.sourceGroup || frame.batchId || frame.jobId ? (
              <p className="mt-1 text-xs text-slate-500">
                {[frame.sourceGroup, frame.batchId, frame.jobId].filter(Boolean).join(' / ')}
              </p>
            ) : null}
            <p className="mt-1 text-sm text-slate-500">Query relevance: {formatQueryRelevance(frame.queryRelevance)}</p>
            {frame.selectionReason ? (
              <p className="mt-2 inline-flex rounded-lg border border-blue-100 bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-900">
                Selected because {frame.selectionReason}
              </p>
            ) : null}
            {detectorBoxFallbackUsed ? (
              <p className="mt-2 inline-flex rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-900">
                Detector-box fallback used
              </p>
            ) : null}
            {useCaseProfile ? (
              <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs leading-5 text-slate-700">
                <span className="font-semibold text-slate-900">{useCaseProfile.label}</span>
                <span className="ml-2 rounded-full border border-slate-200 bg-white px-2 py-0.5 font-semibold uppercase text-slate-500">
                  {supportLevelLabel(useCaseProfile.backendSupportLevel)}
                </span>
                <p>Effective query: <span className="font-semibold">{effectiveQuery ?? useCaseProfile.effectiveQuery ?? 'Not recorded'}</span></p>
                <p>Detector classes involved: {detectorSummary}</p>
                {hiddenProfileViolationCount ? (
                  <p className="font-semibold text-amber-800">
                    {hiddenProfileViolationCount} finding{hiddenProfileViolationCount === 1 ? '' : 's'} outside this profile were de-prioritized from the prominent findings panel.
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>
          <StatusBadge
            label={hasViolations ? 'Violations detected' : 'No violations detected'}
            tone={hasViolations ? 'danger' : 'success'}
          />
        </div>
      </div>

      <div className="grid gap-4 p-4 2xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <EvidenceFrameVisual frame={frame} />
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          {hasViolations ? (
            <div>
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-950">
                <ShieldAlert className="h-4 w-4 text-red-600" aria-hidden="true" />
                Frame findings
              </div>
              <div className="flex flex-col gap-2">
                {displayedViolations.map((violation) => (
                  <div
                    key={violation.id}
                    className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-slate-950">{friendlyFindingName(violation)}</span>
                      <SeverityBadge severity={violation.severity} />
                    </div>
                    <p className="mt-1 text-xs font-medium text-slate-500">
                      Review confidence: {formatConfidence(violation.confidence)}
                    </p>
                    <p className="mt-1 text-xs font-semibold text-slate-600">
                      Review level: {evidenceStrengthLabel(violation.evidenceStrength)}
                      {violation.reviewRequired ? ' - review required' : ''}
                    </p>
                    {violation.confidenceReason ? (
                      <p className="mt-1 text-xs leading-5 text-slate-500">
                        Reason: {violation.confidenceReason}
                      </p>
                    ) : null}
                    {verifierAgreementLabel(violation.verifierAgreement) ? (
                      <p
                        className={clsx(
                          'mt-1 rounded-md border px-2 py-1 text-xs font-semibold',
                          violation.verifierAgreement === 'disagrees'
                            ? 'border-amber-200 bg-amber-50 text-amber-900'
                            : 'border-slate-200 bg-white text-slate-700',
                        )}
                      >
                        {verifierAgreementLabel(violation.verifierAgreement)}
                        {violation.verifierDisagreementReason ? `: ${violation.verifierDisagreementReason}` : ''}
                        {violation.verifierConfidenceHint ? ` Confidence hint: ${violation.verifierConfidenceHint}` : ''}
                      </p>
                    ) : null}
                    {violation.finalReviewerNote ? (
                      <p className="mt-1 text-xs leading-5 text-slate-500">
                        Reviewer note: {violation.finalReviewerNote}
                      </p>
                    ) : null}
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {profileFindingContext(useCaseProfile, violation.name || violation.type)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm font-medium text-emerald-800">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <span>No matching violations found in this frame.</span>
              </div>
              {hiddenProfileViolationCount ? (
                <p className="mt-2 text-xs leading-5 text-emerald-900">
                  Raw detector/rule output included {hiddenProfileViolationCount} finding{hiddenProfileViolationCount === 1 ? '' : 's'} outside this profile; review technical evidence if needed.
                </p>
              ) : null}
            </div>
          )}

          {showExplanation && hasViolations ? (
            <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm leading-6 text-blue-900">
              <div className="mb-3 rounded-lg border border-blue-200 bg-white/80 px-3 py-2">
                <p className="text-sm font-bold text-blue-950">{explanationInfo.header}</p>
                <p className="mt-1 text-xs leading-5 text-blue-800">{explanationInfo.note}</p>
              </div>
              <div className="mb-1 flex items-center gap-2 font-semibold">
                <Sparkles className="h-4 w-4" aria-hidden="true" />
                Safety review
              </div>
              <p className="font-semibold">
                {reviewLevelLabel(primaryViolation?.evidenceStrength)}: possible {friendlyFindingName(primaryViolation)}.
              </p>
              <p className="mt-1">
                Treat this as a review cue, not a final decision.
              </p>
              <div className="mt-3 space-y-3">
                <section>
                  <p className="text-xs font-semibold uppercase tracking-wide text-blue-950">Why this was flagged</p>
                  <p>{whyFlaggedText(primaryViolation)}</p>
                </section>
                <section>
                  <p className="text-xs font-semibold uppercase tracking-wide text-blue-950">Visual review</p>
                  <p>{visualReviewCopy}</p>
                </section>
                <section>
                  <p className="text-xs font-semibold uppercase tracking-wide text-blue-950">Review level</p>
                  <p>{reviewLevelCopy}</p>
                  {primaryViolation?.finalReviewerNote ? (
                    <p className="mt-1">{primaryViolation.finalReviewerNote}</p>
                  ) : null}
                </section>
                <section>
                  <p className="text-xs font-semibold uppercase tracking-wide text-blue-950">What to check in the original footage</p>
                  <p>{whatToCheckText(primaryViolation)}</p>
                </section>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="border-t border-slate-200 p-4">
        <TechnicalDetails frame={frame} jobId={jobId} />
      </div>
    </article>
  );
}
