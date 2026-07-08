import { mockAnalysisByMediaId, mockMediaLibrary, sampleMedia } from '../data/mockAnalysis';
import type {
  AnalysisJob,
  AnalysisRequest,
  AnalysisResult,
  AnalysisSettings,
  BatchAnalysisRequest,
  BatchStatus,
  BackendConnectionState,
  BackendHealth,
  Detection,
  DeviceMode,
  JobStatus,
  MediaItem,
  MediaType,
  Severity,
  SystemStatus,
  UseCaseProfileSelection,
  Violation,
  ViolationEvent,
  VlmExplanationProfileId,
} from '../types/analysis';
import { resolveUseCaseProfile } from '../data/useCaseProfiles';

const MOCK_DELAY_MS = 150;
const DEFAULT_API_BASE = '/api';
const LOCAL_RUNTIME_BASE_CANDIDATES = [
  'http://127.0.0.1:8000',
  'http://localhost:8000',
];
const DISCOVERY_TIMEOUT_MS = 2500;

export const SAFETRACE_REQUIRE_BACKEND = parseEnvBoolean(
  import.meta.env.VITE_SAFETRACE_REQUIRE_BACKEND,
  true,
);
export const SAFETRACE_ENABLE_PREVIEW_MODE = parseEnvBoolean(
  import.meta.env.VITE_SAFETRACE_ENABLE_PREVIEW_MODE,
  false,
);

type RunMockAnalysisInput = {
  query: string;
  media: MediaItem;
  settings: AnalysisSettings;
};

export type BackendDiscoveryResult = {
  apiBase: string;
  health: BackendHealth;
  systemStatus: SystemStatus;
};

export class BackendDiscoveryError extends Error {
  state: BackendConnectionState;
  attemptedBases: string[];

  constructor(message: string, state: BackendConnectionState, attemptedBases: string[]) {
    super(message);
    this.name = 'BackendDiscoveryError';
    this.state = state;
    this.attemptedBases = attemptedBases;
  }
}

type BackendMedia = {
  id: string;
  name: string;
  type: MediaType;
  sizeBytes: number;
  durationSeconds?: number | null;
};

type BackendViolation = {
  id: string;
  name: string;
  severity: string;
  confidence: number;
  description: string;
  evidence?: Record<string, unknown>;
  evidenceStrength?: string;
  confidenceReason?: string;
  reviewRequired?: boolean;
  ruleSupport?: string;
  suppressedFindings?: unknown[];
  unsupportedRuleReason?: string | null;
  verifierAgreement?: string | null;
  verifierDisagreementReason?: string | null;
  verifierConfidenceHint?: string | null;
  finalReviewerNote?: string | null;
};

type BackendGroupedViolation = {
  id: string;
  name: string;
  severity: string;
  description: string;
  affectedFrames: Array<{
    frameId: string;
    frameNumber: number;
    timestamp: string;
    confidence: number;
  }>;
  confidenceMin: number;
  confidenceMax: number;
};

type BackendFrameResult = {
  id: string;
  frameNumber: number;
  timestamp: string;
  queryRelevance: number;
  status: 'violations_detected' | 'no_violations';
  imageUrl?: string | null;
  imageMessage?: string | null;
  explanationSource?: 'vlm' | 'vlm_local' | 'vlm_ollama' | 'vlm_lightweight' | 'vlm_enhanced' | 'rule_template_plus_vlm' | 'rule_template_plus_lightweight_vlm' | 'rule_template_plus_lightweight_plus_enhanced' | 'rule_based' | null;
  violations: BackendViolation[];
  technicalEvidence: Record<string, unknown>;
};

type BackendViolationEvent = {
  id: string;
  type: string;
  name: string;
  severity: string;
  description: string;
  startTimestamp: string;
  endTimestamp: string;
  representativeConfidence: number;
  confidenceMin: number;
  confidenceMax: number;
  supportingFrameCount: number;
  supportingFrames: Array<{
    frameId: string;
    frameNumber: number;
    timestamp: string;
    confidence: number;
    imageUrl?: string | null;
  }>;
};

type BackendAnalysisResult = {
  jobId: string;
  status: 'completed';
  media: BackendMedia;
  query: string;
  summary: {
    framesAnalyzed: number;
    framesWithViolations: number;
    uniqueViolationTypes: number;
    highestSeverity?: string | null;
    summaryText: string;
    potentialEventCount?: number;
    eventTypes?: string[];
    overallConfidence?: number;
    keyEvents?: unknown[];
  };
  violations: BackendGroupedViolation[];
  events?: BackendViolationEvent[];
  frames: BackendFrameResult[];
  technicalDetails?: Record<string, unknown> | null;
};

function serializeUseCaseProfile(profile?: UseCaseProfileSelection): string | undefined {
  if (!profile) return undefined;
  return JSON.stringify({
    profileId: profile.profileId,
    label: profile.label,
    category: profile.category,
    description: profile.description,
    defaultQuery: profile.defaultQuery,
    backendSupportLevel: profile.backendSupportLevel,
    supportedChecks: profile.supportedChecks ?? [],
    unsupportedChecks: profile.unsupportedChecks ?? [],
    limitations: profile.limitations,
    checks: profile.checks ?? [],
    rules: profile.rules ?? [],
    notes: profile.notes,
    customText: profile.customText,
    requestedQuery: profile.requestedQuery,
    effectiveQuery: profile.effectiveQuery,
  });
}

function parseEnvBoolean(value: string | boolean | undefined, fallback: boolean): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value !== 'string') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, '');
const trimLeadingSlash = (value: string) => value.replace(/^\/+/, '');

function isAbsoluteHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function normalizeApiBase(value: string): string {
  const trimmed = trimTrailingSlash(value.trim());
  if (!trimmed) return DEFAULT_API_BASE;
  if (trimmed === DEFAULT_API_BASE || trimmed.endsWith('/api')) return trimmed;
  if (trimmed === '/') return DEFAULT_API_BASE;
  return `${trimmed}/api`;
}

function configuredApiBase(): string | null {
  const value = import.meta.env.VITE_SAFETRACE_API_BASE_URL || import.meta.env.VITE_SAFETRACE_API_BASE;
  return typeof value === 'string' && value.trim() ? normalizeApiBase(value) : null;
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

function buildApiBaseCandidates(): string[] {
  const configured = configuredApiBase();
  if (configured) return [configured];
  return unique(LOCAL_RUNTIME_BASE_CANDIDATES.map(normalizeApiBase));
}

export const SAFETRACE_API_BASE_CANDIDATES = buildApiBaseCandidates();
let activeApiBase = SAFETRACE_API_BASE_CANDIDATES[0] || DEFAULT_API_BASE;

export const SAFETRACE_API_BASE = activeApiBase;

export function getActiveApiBase(): string {
  return activeApiBase;
}

function setActiveApiBase(apiBase: string): void {
  activeApiBase = normalizeApiBase(apiBase);
}

export function buildApiUrl(path: string, apiBase = activeApiBase): string {
  const base = trimTrailingSlash(apiBase || DEFAULT_API_BASE);
  const cleanPath = trimLeadingSlash(path);
  return `${base}/${cleanPath}`;
}

export function getApiOrigin(): string {
  if (isAbsoluteHttpUrl(activeApiBase)) {
    return new URL(activeApiBase).origin;
  }
  return window.location.origin;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function requestTimeoutSignal(timeoutMs: number): AbortSignal {
  const controller = new AbortController();
  window.setTimeout(() => controller.abort(), timeoutMs);
  return controller.signal;
}

function apiErrorMessage(status: number): string {
  if (status === 404) return 'SafeTrace API route was not found on the local runtime.';
  if (status >= 500) return 'SafeTrace Local Runtime responded with a server error.';
  return `SafeTrace API returned ${status}`;
}

export async function readErrorMessage(response: Response): Promise<string> {
  const fallback = apiErrorMessage(response.status);
  const raw = await response.text();
  if (!raw) return response.statusText ? `${fallback}: ${response.statusText}` : fallback;

  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === 'string') return parsed.detail;
    if (typeof parsed?.detail?.message === 'string') return parsed.detail.message;
    if (typeof parsed?.message === 'string') return parsed.message;
    if (typeof parsed?.error === 'string') return parsed.error;
  } catch {
    return raw;
  }

  return raw || fallback;
}

async function apiFetchFromBase<T>(apiBase: string, path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(buildApiUrl(path, apiBase), {
    ...options,
    headers: {
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<T>;
}

async function probeApiBase(apiBase: string): Promise<BackendDiscoveryResult> {
  const health = await apiFetchFromBase<BackendHealth>(apiBase, 'health', {
    cache: 'no-store',
    signal: requestTimeoutSignal(DISCOVERY_TIMEOUT_MS),
  });
  if (health.status !== 'ok' || health.api !== 'safetrace-local') {
    throw new BackendDiscoveryError(
      `A service responded at ${apiBase}, but it is not the SafeTrace Local Runtime.`,
      'incompatible',
      [apiBase],
    );
  }
  const systemStatus = await apiFetchFromBase<SystemStatus>(apiBase, 'system/status', {
    cache: 'no-store',
    signal: requestTimeoutSignal(DISCOVERY_TIMEOUT_MS),
  });
  setActiveApiBase(apiBase);
  return { apiBase: getActiveApiBase(), health, systemStatus };
}

export async function discoverBackendRuntime(): Promise<BackendDiscoveryResult> {
  const attempted = [...SAFETRACE_API_BASE_CANDIDATES];
  const messages: string[] = [];
  let incompatibleMessage: string | null = null;

  for (const candidate of attempted) {
    try {
      return await probeApiBase(candidate);
    } catch (err) {
      if (err instanceof BackendDiscoveryError && err.state === 'incompatible') {
        incompatibleMessage = err.message;
      }
      if (err instanceof Error && err.name === 'AbortError') {
        messages.push(`${candidate}: connection timed out`);
      } else if (err instanceof Error) {
        messages.push(`${candidate}: ${err.message}`);
      } else {
        messages.push(`${candidate}: local runtime did not respond`);
      }
    }
  }

  if (incompatibleMessage) {
    throw new BackendDiscoveryError(incompatibleMessage, 'incompatible', attempted);
  }

  throw new BackendDiscoveryError(
    `SafeTrace Local Runtime not connected. Tried ${attempted.join(', ')}.`,
    'disconnected',
    attempted,
  );
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  return apiFetchFromBase<T>(activeApiBase, path, options);
}

function cloneResult(result: AnalysisResult): AnalysisResult {
  return {
    ...result,
    media: { ...result.media },
    frames: result.frames.map((frame) => ({
      ...frame,
      violations: frame.violations.map((violation) => ({
        ...violation,
        evidence: violation.evidence ? { ...violation.evidence } : undefined,
      })),
      detections: frame.detections.map((detection) => ({ ...detection })),
      technicalEvidence: { ...frame.technicalEvidence },
    })),
    events: result.events?.map((event) => ({
      ...event,
      supportingFrames: event.supportingFrames.map((frame) => ({ ...frame })),
    })),
  };
}

function toSeverity(value: string | undefined): Severity {
  const normalized = (value || '').toLowerCase();
  if (normalized === 'high' || normalized === 'critical') return 'High';
  if (normalized === 'medium') return 'Medium';
  return 'Low';
}

function toMediaType(value: string | undefined): MediaType {
  if (value === 'video' || value === 'image') return value;
  return 'unknown';
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'Unknown size';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let size = bytes / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

function formatDuration(seconds?: number | null): string | undefined {
  if (!seconds || !Number.isFinite(seconds)) return undefined;
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function getDetectionBBox(value: unknown): [number, number, number, number] {
  if (Array.isArray(value) && value.length >= 4) {
    const [x, y, width, height] = value.map(Number);
    if ([x, y, width, height].every(Number.isFinite)) {
      return [x, y, width, height];
    }
  }
  return [0, 0, 0, 0];
}

function getDetectionValue(detection: unknown, key: string): unknown {
  if (typeof detection === 'object' && detection !== null && key in detection) {
    return (detection as Record<string, unknown>)[key];
  }
  return undefined;
}

function getRecordValue(value: unknown, key: string): unknown {
  if (typeof value === 'object' && value !== null && key in value) {
    return (value as Record<string, unknown>)[key];
  }
  return undefined;
}

function getNestedRecordValue(value: unknown, path: string[]): unknown {
  return path.reduce<unknown>((current, key) => getRecordValue(current, key), value);
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function mapUseCaseProfile(value: unknown): UseCaseProfileSelection | undefined {
  if (typeof value !== 'object' || value === null) return undefined;
  const profileId = optionalString(getRecordValue(value, 'profileId'));
  if (!profileId) return undefined;
  const customText = optionalString(getRecordValue(value, 'customText')) || optionalString(getRecordValue(value, 'notes')) || '';
  const requestedQuery = optionalString(getRecordValue(value, 'requestedQuery')) || optionalString(getRecordValue(value, 'effectiveQuery')) || '';
  const resolved = resolveUseCaseProfile(profileId, customText, requestedQuery);
  return {
    ...resolved,
    requestedQuery: requestedQuery || resolved.requestedQuery,
    effectiveQuery: optionalString(getRecordValue(value, 'effectiveQuery')) || resolved.effectiveQuery,
  };
}

function mapDetections(frame: BackendFrameResult): Detection[] {
  const rawDetections = frame.technicalEvidence?.detections;
  if (!Array.isArray(rawDetections)) return [];

  return rawDetections.map((detection, index) => {
    const label = String(getDetectionValue(detection, 'label') || getDetectionValue(detection, 'name') || 'detection');
    const confidence = Number(getDetectionValue(detection, 'confidence') || 0);

    return {
      id: `${frame.id}-detection-${index}`,
      label,
      confidence: Number.isFinite(confidence) ? confidence : 0,
      bbox: getDetectionBBox(getDetectionValue(detection, 'bbox')),
      source: 'detector',
    };
  });
}

function mapViolations(violations: BackendViolation[]): Violation[] {
  return violations.map((violation) => ({
    id: violation.id,
    type: violation.id,
    name: violation.name,
    severity: toSeverity(violation.severity),
    description: violation.description,
    confidence: violation.confidence,
    evidence: violation.evidence ? { ...violation.evidence } : undefined,
    evidenceStrength: violation.evidenceStrength,
    confidenceReason: violation.confidenceReason,
    reviewRequired: Boolean(violation.reviewRequired),
    ruleSupport: violation.ruleSupport,
    suppressedFindings: violation.suppressedFindings,
    unsupportedRuleReason: violation.unsupportedRuleReason,
    verifierAgreement: violation.verifierAgreement,
    verifierDisagreementReason: violation.verifierDisagreementReason,
    verifierConfidenceHint: violation.verifierConfidenceHint,
    finalReviewerNote: violation.finalReviewerNote,
  }));
}

function violationReviewLine(violation: Violation): string {
  const key = `${violation.type} ${violation.name}`.toLowerCase();
  if (key.includes('helmet')) {
    return "Helmet: the worker's head is visible, but no helmet was detected over it.";
  }
  if (key.includes('seatbelt') || key.includes('seat belt')) {
    return "Seatbelt: the worker's torso is visible, but no seatbelt was detected across it.";
  }
  if (key.includes('phone')) {
    return 'Phone use: a phone appears close to a detected hand and should be reviewed.';
  }
  if (key.includes('wheel') || key.includes('hand')) {
    return 'Hands on controls: detected hands do not appear to be on the expected steering/control area.';
  }
  if (key.includes('restricted') || key.includes('zone')) {
    return 'Restricted area: a person appears inside or close to a monitored restricted zone.';
  }
  if (key.includes('vest')) {
    return 'High-visibility vest: the person is visible, but vest evidence appears weak or missing.';
  }
  return `${violation.name}: review this visible finding against the original footage.`;
}

const VLM_ARTIFACT_PATTERN = /<(?:global-[a-z]+|image|row_[^>\s]*|[^>\s]*(?:img|row_|col_)[^>\s]*)>/i;
const VLM_ROLE_LABEL_PATTERN = /\b(?:user|assistant)\s*:/i;
const VLM_PROMPT_ECHO_PATTERN = /describe only visible safety evidence|return only the final explanation|do not repeat the prompt|findings to inspect/i;

function explanationLooksTechnical(value: string): boolean {
  if (VLM_ARTIFACT_PATTERN.test(value)) return true;
  if (VLM_ROLE_LABEL_PATTERN.test(value)) return true;
  if (VLM_PROMPT_ECHO_PATTERN.test(value)) return true;
  return /\b(iou|threshold|overlap|raw|internal|configured|metric|count|minimum|maximum|score|key)\b/i.test(value);
}

function buildVisualExplanation(violations: Violation[], rawExplanation: unknown): string | undefined {
  const explanation = typeof rawExplanation === 'string' ? rawExplanation.trim() : '';
  if (!violations.length) {
    return explanation && !explanationLooksTechnical(explanation)
      ? explanation
      : 'SafeTrace did not find a matching safety issue in this frame. Review the original footage if camera angle, blur, glare, or shadows could hide important details.';
  }
  if (explanation && !explanationLooksTechnical(explanation)) return explanation;

  const reviewLines = Array.from(new Set(violations.map(violationReviewLine)));
  return [
    'SafeTrace flagged this frame because visible scene evidence matches the selected safety query.',
    '',
    'What to review:',
    ...reviewLines.map((line) => `- ${line}`),
    '',
    'Reviewer note: confirm the finding is not caused by camera angle, blur, glare, shadows, or hidden equipment.',
  ].join('\n');
}

function explanationSourceFor(
  rawSource: unknown,
  rawExplanation: unknown,
  violations: Violation[],
): 'vlm' | 'vlm_local' | 'vlm_ollama' | 'vlm_lightweight' | 'vlm_enhanced' | 'rule_template_plus_vlm' | 'rule_template_plus_lightweight_vlm' | 'rule_template_plus_lightweight_plus_enhanced' | 'rule_based' | undefined {
  const source = String(rawSource || '').toLowerCase();
  const explanation = typeof rawExplanation === 'string' ? rawExplanation.trim() : '';
  if (
    (
      source === 'rule_template_plus_vlm'
      || source === 'rule_template_plus_lightweight_vlm'
      || source === 'rule_template_plus_lightweight_plus_enhanced'
    )
    && explanation
  ) {
    return source;
  }
  if ((source === 'vlm_lightweight' || source === 'vlm_enhanced') && explanation && !explanationLooksTechnical(explanation)) {
    return source;
  }
  if ((source === 'vlm_local' || source === 'vlm_ollama') && explanation && !explanationLooksTechnical(explanation)) {
    return source;
  }
  if (source === 'vlm' && explanation && !explanationLooksTechnical(explanation)) return 'vlm';
  if (violations.length || explanation) return 'rule_based';
  return undefined;
}

function mapEvents(events: BackendViolationEvent[] | undefined): ViolationEvent[] | undefined {
  if (!events) return undefined;
  return events.map((event) => ({
    id: event.id,
    type: event.type,
    name: event.name,
    severity: toSeverity(event.severity),
    description: event.description,
    startTimestamp: event.startTimestamp,
    endTimestamp: event.endTimestamp,
    representativeConfidence: event.representativeConfidence,
    confidenceMin: event.confidenceMin,
    confidenceMax: event.confidenceMax,
    supportingFrameCount: event.supportingFrameCount,
    supportingFrames: event.supportingFrames.map((frame) => ({
      ...frame,
      imageUrl: resolveBackendMediaUrl(frame.imageUrl ?? null),
    })),
  }));
}

function mapBackendResult(result: BackendAnalysisResult): AnalysisResult {
  const mediaName = result.media.name || result.media.id || 'Selected media';
  const useCaseProfile = mapUseCaseProfile(
    getNestedRecordValue(result.technicalDetails, ['jobMetrics', 'componentDiagnostics', 'useCaseProfile'])
      ?? getNestedRecordValue(result.technicalDetails, ['jobMetrics', 'analysisSettings', 'useCaseProfile'])
      ?? getNestedRecordValue(result.technicalDetails, ['processingMetadata', 'useCaseProfile']),
  );

  return {
    jobId: result.jobId,
    status: result.status,
    id: result.jobId,
    query: result.query,
    media: {
      id: result.media.id || `media-${result.jobId}`,
      filename: mediaName,
      type: toMediaType(result.media.type),
      sizeLabel: formatBytes(result.media.sizeBytes),
      duration: formatDuration(result.media.durationSeconds),
      uploadedAt: new Date().toISOString(),
      status: 'completed',
      source: 'local',
      jobId: result.jobId,
      selectedJobId: result.jobId,
      useCaseProfile,
    },
    summary: result.summary,
    violations: result.violations,
    events: mapEvents(result.events),
    framesAnalyzed: result.summary.framesAnalyzed,
    generatedAt: new Date().toISOString(),
    summaryText: result.summary.summaryText,
    settings: useCaseProfile ? {
      fps: 1,
      topK: result.frames.length,
      visualExplanations: true,
      vlmProfile: 'rule_based',
      vlmEnabled: false,
      enhancedVlmExplanations: false,
      deviceMode: 'Auto',
      useCaseProfile,
    } : undefined,
    frames: result.frames.map((frame) => {
      const violations = mapViolations(frame.violations);
      const rawExplanation = frame.technicalEvidence?.explanation;
      const rawExplanationSource = frame.explanationSource ?? frame.technicalEvidence?.explanationSource;
      const searchMetadata = frame.technicalEvidence?.searchMetadata;
      const rawFrameScore = Number(getRecordValue(searchMetadata, 'frameRankingScore'));
      return {
        id: frame.id,
        frameNumber: frame.frameNumber,
        timestamp: frame.timestamp,
        internalFilename: String(frame.technicalEvidence?.sourceFramePath || frame.id),
        queryRelevance: frame.queryRelevance,
        imageUrl: resolveBackendMediaUrl(frame.imageUrl ?? null) ?? undefined,
        imageMessage: frame.imageMessage ?? undefined,
        evidenceImageRequired: Boolean(frame.imageUrl || frame.imageMessage),
        selectionReason: optionalString(getRecordValue(searchMetadata, 'rankingReason')),
        selectionCategory: optionalString(getRecordValue(searchMetadata, 'selectedFor')),
        frameScore: Number.isFinite(rawFrameScore) ? rawFrameScore : undefined,
        explanation: buildVisualExplanation(violations, rawExplanation),
        explanationSource: explanationSourceFor(rawExplanationSource, rawExplanation, violations),
        violations,
        detections: mapDetections(frame),
        technicalEvidence: frame.technicalEvidence,
      };
    }),
    technicalDetails: result.technicalDetails,
  };
}

function deviceToApiMode(device: DeviceMode): 'auto' | 'cpu' | 'cuda' {
  if (device === 'CPU') return 'cpu';
  if (device === 'GPU') return 'cuda';
  return 'auto';
}

export function buildMockAnalysisResult({
  query,
  media,
  settings,
}: RunMockAnalysisInput): AnalysisResult {
  const template = mockAnalysisByMediaId[media.id] ?? mockAnalysisByMediaId[sampleMedia.id];
  if (!template) {
    return {
      id: 'analysis-empty',
      query,
      media,
      framesAnalyzed: 0,
      generatedAt: new Date().toISOString(),
      summaryText: 'No matching safety violations were detected.',
      frames: [],
      settings,
    };
  }
  const result = cloneResult(template);
  const limitedFrames = result.frames.slice(0, settings.topK).map((frame) => {
    const friendlyFrame = {
      ...frame,
      explanation: buildVisualExplanation(frame.violations, frame.explanation),
      explanationSource: frame.explanationSource ?? 'rule_based',
    };
    if (media.source === 'local' && media.type === 'image' && media.previewUrl) {
      return { ...friendlyFrame, imageUrl: media.previewUrl, evidenceImageRequired: false };
    }
    return friendlyFrame;
  });

  return {
    ...result,
    query: query.trim() || result.query,
    media: { ...media, status: 'completed' },
    framesAnalyzed: limitedFrames.length,
    frames: limitedFrames,
    generatedAt: new Date().toISOString(),
    settings,
  };
}

export async function runMockAnalysis(input: RunMockAnalysisInput): Promise<AnalysisResult> {
  await delay(MOCK_DELAY_MS);
  return buildMockAnalysisResult(input);
}

export async function getMockMediaLibrary(): Promise<MediaItem[]> {
  await delay(250);
  return mockMediaLibrary;
}

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

export async function checkBackendHealth(): Promise<BackendHealth> {
  const result = await discoverBackendRuntime();
  return result.health;
}

export async function getSystemStatus(): Promise<SystemStatus> {
  return apiFetch<SystemStatus>('system/status');
}

export async function updateVlmSettings(settings: {
  selectedProfile: VlmExplanationProfileId;
  enabled: boolean;
}): Promise<boolean> {
  const response = await fetch(buildApiUrl('system/vlm/settings'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(settings),
  });

  if (response.status === 404 || response.status === 405) return false;

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return true;
}

export async function runBackendAnalysis(request: AnalysisRequest): Promise<AnalysisJob> {
  const formData = new FormData();
  formData.append('file', request.file);
  formData.append('query', request.query);
  formData.append('fps', String(request.fps));
  formData.append('topK', String(request.topK));
  formData.append('enableVlm', String(request.enableVlm));
  if (request.vlmProfile) formData.append('vlmProfile', request.vlmProfile);
  if (typeof request.vlmEnabled === 'boolean') formData.append('vlmEnabled', String(request.vlmEnabled));
  const useCaseProfile = serializeUseCaseProfile(request.useCaseProfile);
  if (useCaseProfile) formData.append('useCaseProfile', useCaseProfile);
  formData.append('device', deviceToApiMode(request.device));

  return apiFetch<AnalysisJob>('analyze', {
    method: 'POST',
    body: formData,
  });
}

export async function runBackendBatchAnalysis(request: BatchAnalysisRequest): Promise<BatchStatus> {
  const formData = new FormData();
  request.files.forEach((file) => {
    formData.append('files', file);
  });
  formData.append('query', request.query);
  formData.append('fps', String(request.fps));
  formData.append('topK', String(request.topK));
  formData.append('enableVlm', String(request.enableVlm));
  if (request.vlmProfile) formData.append('vlmProfile', request.vlmProfile);
  if (typeof request.vlmEnabled === 'boolean') formData.append('vlmEnabled', String(request.vlmEnabled));
  const useCaseProfile = serializeUseCaseProfile(request.useCaseProfile);
  if (useCaseProfile) formData.append('useCaseProfile', useCaseProfile);
  formData.append('device', deviceToApiMode(request.device));

  return apiFetch<BatchStatus>('batches/analyze', {
    method: 'POST',
    body: formData,
  });
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return apiFetch<JobStatus>(`jobs/${encodeURIComponent(jobId)}`);
}

export async function getBatchStatus(batchId: string): Promise<BatchStatus> {
  return apiFetch<BatchStatus>(`batches/${encodeURIComponent(batchId)}`);
}

export async function getJobResult(jobId: string): Promise<AnalysisResult> {
  const result = await apiFetch<BackendAnalysisResult>(`jobs/${encodeURIComponent(jobId)}/result`);
  return mapBackendResult(result);
}

export async function getTechnicalReport(jobId: string): Promise<AnalysisResult> {
  const result = await apiFetch<BackendAnalysisResult>(`reports/${encodeURIComponent(jobId)}/technical-json`);
  return mapBackendResult(result);
}

export async function deleteJob(jobId: string): Promise<void> {
  await apiFetch(`jobs/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
  });
}

export async function deleteBatch(batchId: string): Promise<void> {
  await apiFetch(`batches/${encodeURIComponent(batchId)}`, {
    method: 'DELETE',
  });
}

export function resolveBackendMediaUrl(imageUrl: string | null): string | null {
  if (!imageUrl) return null;
  if (/^(https?:|blob:|data:)/i.test(imageUrl)) return imageUrl;

  const base = trimTrailingSlash(activeApiBase || DEFAULT_API_BASE);
  const isAbsoluteApiBase = isAbsoluteHttpUrl(base);
  const origin = getApiOrigin();

  if (imageUrl.startsWith('/api/')) {
    return isAbsoluteApiBase ? `${origin}${imageUrl}` : imageUrl;
  }
  if (imageUrl.startsWith('/media/')) {
    return `${base}${imageUrl}`;
  }
  if (imageUrl.startsWith('/')) {
    return isAbsoluteApiBase ? `${origin}${imageUrl}` : imageUrl;
  }

  const cleanPath = trimLeadingSlash(imageUrl);
  if (cleanPath.startsWith('api/')) {
    return isAbsoluteApiBase ? `${origin}/${cleanPath}` : `/${cleanPath}`;
  }
  return `${base}/${cleanPath}`;
}
