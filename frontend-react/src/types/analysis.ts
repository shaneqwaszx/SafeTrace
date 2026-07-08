export type Severity = 'High' | 'Medium' | 'Low';

export type MediaType = 'video' | 'image' | 'unknown';

export type MediaStatus = 'draft' | 'ready' | 'queued' | 'processing' | 'completed' | 'error';

export type DeviceMode = 'Auto' | 'CPU' | 'GPU';

export type BackendConnectionState = 'live' | 'connecting' | 'connected' | 'disconnected' | 'incompatible' | 'error';

export type VlmExplanationProfileId =
  | 'rule_based'
  | 'lightweight_256m'
  | 'lightweight_512m'
  | 'enhanced_2b'
  | 'enhanced_3b';

export type UseCaseProfileId =
  | 'seatbelt_compliance'
  | 'phone_use'
  | 'helmet_ppe'
  | 'uniform_compliance'
  | 'general_safety'
  | 'custom_policy';

export type BackendSupportLevel = 'supported' | 'partial' | 'metadata_only' | 'unsupported';

export type UseCaseProfileSelection = {
  profileId: UseCaseProfileId;
  label: string;
  category: string;
  description: string;
  defaultQuery: string;
  supportedChecks: string[];
  unsupportedChecks?: string[];
  limitations?: string;
  backendSupportLevel: BackendSupportLevel;
  checks?: string[];
  rules?: string[];
  notes?: string;
  customText?: string;
  requestedQuery?: string;
  effectiveQuery?: string;
};

export type AnalysisSettings = {
  fps: number;
  topK: number;
  visualExplanations: boolean;
  vlmProfile: VlmExplanationProfileId;
  vlmEnabled: boolean;
  enhancedVlmExplanations?: boolean;
  vlmExplanations?: boolean;
  deviceMode: DeviceMode;
  useCaseProfile: UseCaseProfileSelection;
};

export type Violation = {
  id: string;
  type: string;
  name: string;
  severity: Severity;
  description: string;
  confidence: number;
  evidence?: Record<string, unknown>;
  evidenceStrength?: 'confirmed_violation' | 'likely_violation' | 'review_candidate' | 'insufficient_evidence' | 'unsupported_rule' | string;
  confidenceReason?: string;
  reviewRequired?: boolean;
  ruleSupport?: string;
  suppressedFindings?: unknown[];
  unsupportedRuleReason?: string | null;
  verifierAgreement?: 'supports' | 'inconclusive' | 'disagrees' | string | null;
  verifierDisagreementReason?: string | null;
  verifierConfidenceHint?: string | null;
  finalReviewerNote?: string | null;
};

export type Detection = {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  source: 'detector' | 'rule-engine' | 'mock';
};

export type FrameResult = {
  id: string;
  frameNumber: number;
  timestamp: string;
  internalFilename: string;
  queryRelevance: number;
  queryRelevanceScore?: number;
  score?: number;
  imageUrl?: string;
  imageMessage?: string;
  evidenceImageRequired?: boolean;
  selectionReason?: string;
  selectionCategory?: string;
  frameScore?: number;
  visualVariant?: 'worksite' | 'loading-bay' | 'maintenance';
  explanation?: string;
  explanationSource?: 'vlm' | 'vlm_local' | 'vlm_ollama' | 'vlm_lightweight' | 'vlm_enhanced' | 'rule_template_plus_vlm' | 'rule_template_plus_lightweight_vlm' | 'rule_template_plus_lightweight_plus_enhanced' | 'rule_based';
  violations: Violation[];
  detections: Detection[];
  technicalEvidence: Record<string, unknown>;
};

export type Annotation = {
  id: string;
  mediaId: string;
  type: 'bbox' | 'note';
  label?: string;
  note?: string;
  bbox?: [number, number, number, number];
  color?: string;
  createdAt: string;
};

export type MediaItem = {
  id: string;
  filename: string;
  type: MediaType;
  sizeLabel: string;
  duration?: string;
  uploadedAt: string;
  status: MediaStatus;
  source?: 'sample' | 'local';
  previewUrl?: string;
  jobId?: string;
  selectedJobId?: string;
  batchId?: string;
  useCaseProfile?: UseCaseProfileSelection;
  requestedQuery?: string;
  effectiveQuery?: string;
  errorMessage?: string;
};

export type BackendHealth = {
  status: 'ok';
  api: 'safetrace-local';
  version: string;
  offline: boolean;
};

export type BackendModelStatus = {
  status: 'ready' | 'available' | 'missing' | 'missing_checkpoint' | 'missing_runtime' | 'disabled' | 'unavailable';
  path?: string | null;
  message?: string | null;
  actionHint?: string | null;
  details?: Record<string, unknown>;
};

export type RuntimeCheckStatus = 'ready' | 'available' | 'warning' | 'missing' | 'missing_checkpoint' | 'missing_runtime' | 'unavailable' | 'disabled' | 'loading';

export type RuntimeCheck = {
  status: RuntimeCheckStatus | string;
  message?: string | null;
  path?: string | null;
  actionHint?: string | null;
  details?: Record<string, unknown>;
};

export type VlmProfileStatus = {
  id: VlmExplanationProfileId | string;
  label: string;
  installed?: boolean;
  available?: boolean;
  requiresActivation?: boolean;
  resourceLevel?: string;
  path?: string | null;
  message?: string | null;
  statusCopy?: string | null;
  deprecated?: boolean;
  notViable?: boolean;
  candidate?: boolean;
  legacy?: boolean;
  fallback?: boolean;
  requiresGpu?: boolean;
  deviceDecision?: Record<string, unknown> | null;
};

export type SystemVlmStatus = {
  selectedProfile?: VlmExplanationProfileId | string;
  enabled?: boolean;
  active?: boolean;
  runtimeAvailable?: boolean;
  profiles?: VlmProfileStatus[];
  message?: string | null;
  requestedVisualExplanationMode?: string | null;
  actualExplanationMode?: string | null;
  vlmAvailability?: string | null;
  vlmSuppressedReason?: string | null;
  fallbackReason?: string | null;
  lightweightModelPathChecked?: string | null;
  ruleBasedFallbackActive?: boolean | null;
  ruleBasedFallbackAvailable?: boolean | null;
  lightweightVlmWorkerEnabled?: boolean | null;
  lightweightVlmWorkerTimeoutSeconds?: number | null;
  lightweightVlmExplanationSource?: string | null;
  lightweightVlmEvidenceBudget?: number | null;
  lightweightVlmJobTimeoutSeconds?: number | null;
  lightweightVlmMaxQualityFailures?: number | null;
  lightweightVlmPrimaryPolicy?: string | null;
  lightweightVlmFallbackPolicy?: string | null;
  lightweightVlmCpuPrefer256m?: boolean | null;
  enhancedVlmRequiresGpu?: boolean | null;
  deviceGateway?: Record<string, unknown> | null;
};

export type SystemRuntimeStatus = {
  backend?: Record<string, unknown>;
  python?: {
    executable?: string;
    version?: string;
  };
  workingDirectory?: string;
  device?: {
    configured?: string;
    gpuAvailable?: boolean;
    gateway?: Record<string, unknown>;
    detector?: Record<string, unknown>;
    mobileSam?: Record<string, unknown>;
    lightweightVlm?: Record<string, unknown>;
    enhancedVlm?: Record<string, unknown>;
  };
  analysis?: {
    safeMode?: boolean;
    safeModeMobileSamAllowed?: boolean;
    mobileSamWorkerEnabled?: boolean;
    mobileSamWorkerTimeoutSeconds?: number;
    lightweightVlmWorkerEnabled?: boolean;
    lightweightVlmWorkerTimeoutSeconds?: number;
    lightweightVlmEvidenceBudget?: number;
    lightweightVlmJobTimeoutSeconds?: number;
    lightweightVlmMaxQualityFailures?: number;
    effectiveDevice?: string;
    mobileSamDevice?: string | null;
    lightweightVlmDevice?: string | null;
    enhancedVlmDevice?: string | null;
    safeModeMessage?: string | null;
    analysisJobTimeoutSeconds?: number;
  };
  models?: Record<string, BackendModelStatus>;
  chat?: {
    enabled?: boolean;
    available?: boolean;
    state?: string;
    status?: string;
    provider?: string;
    model?: string | null;
    model_path?: string | null;
    model_exists?: boolean | null;
    runtime_available?: boolean | null;
    speed_profile?: string | null;
    warmup_on_open?: boolean | null;
    reason?: string | null;
    action_hint?: string | null;
    message?: string | null;
  };
  openmp?: {
    status?: string;
    kmpDuplicateLibOk?: boolean;
    rawKmpDuplicateLibOk?: string | null;
    ompNumThreads?: string | null;
    message?: string | null;
    actionHint?: string | null;
  };
  uploadLimits?: Record<string, unknown>;
  batchLimits?: Record<string, unknown>;
  jobStorePath?: string;
  visual_explanations?: RuntimeCheck & {
    fallback?: 'rule_based';
    explanationSource?: 'vlm' | 'rule_template_plus_vlm' | 'rule_template_plus_lightweight_vlm' | 'rule_template_plus_lightweight_plus_enhanced' | 'rule_based';
    enhancedVlmAvailable?: boolean;
    requestedVisualExplanationMode?: string | null;
    actualExplanationMode?: string | null;
    fallbackReason?: string | null;
    ruleBasedFallbackActive?: boolean | null;
    lightweightModelPathChecked?: string | null;
    lightweightVlmWorkerEnabled?: boolean | null;
    lightweightVlmWorkerTimeoutSeconds?: number | null;
    lightweightVlmExplanationSource?: string | null;
    lightweightVlmEvidenceBudget?: number | null;
    lightweightVlmJobTimeoutSeconds?: number | null;
    lightweightVlmMaxQualityFailures?: number | null;
  };
};

export type SystemPreflightStatus = {
  checks?: Record<string, RuntimeCheck>;
  summary?: {
    ready?: number;
    warnings?: number;
  };
};

export type SystemStatus = {
  app_version?: string | null;
  backend_version?: string | null;
  build_mode?: string | null;
  runtime_layout?: string | null;
  safeMode?: boolean | null;
  device: string;
  gpuAvailable: boolean;
  models: Record<string, BackendModelStatus>;
  limits?: Record<string, unknown>;
  queue?: Record<string, unknown>;
  runtime?: SystemRuntimeStatus;
  preflight?: SystemPreflightStatus;
  vlm?: SystemVlmStatus;
};

export type AnalysisRequest = {
  file: File;
  query: string;
  fps: number;
  topK: number;
  enableVlm: boolean;
  vlmProfile?: VlmExplanationProfileId;
  vlmEnabled?: boolean;
  device: DeviceMode;
  useCaseProfile?: UseCaseProfileSelection;
};

export type BatchAnalysisRequest = {
  files: File[];
  query: string;
  fps: number;
  topK: number;
  enableVlm: boolean;
  vlmProfile?: VlmExplanationProfileId;
  vlmEnabled?: boolean;
  device: DeviceMode;
  useCaseProfile?: UseCaseProfileSelection;
};

export type AnalysisJob = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
};

export type JobStatus = AnalysisJob & {
  progress: number;
  progressPercent?: number;
  stage?: 'queued' | 'preparing' | 'analyzing' | 'normalizing' | 'completed' | 'failed' | 'cancelled' | string;
  message?: string;
  currentStep: string;
  error?: string | null;
  metrics?: Record<string, unknown>;
  componentDiagnostics?: Record<string, unknown> | null;
  createdAt?: string | null;
  queuedAt?: string | null;
  updatedAt?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  completedAt?: string | null;
  failedAt?: string | null;
  cancelledAt?: string | null;
  elapsedSeconds?: number | null;
  queueWaitSeconds?: number | null;
  analysisRuntimeSeconds?: number | null;
  heartbeatAt?: string | null;
  persistenceWarning?: string | null;
  manifestPersistenceWarning?: string | null;
};

export type BatchAcceptedFile = {
  originalFilename: string;
  filename: string;
  sizeBytes: number;
  mediaType: 'video';
  jobId: string;
  status: AnalysisJob['status'];
  error?: string | null;
};

export type BatchRejectedFile = {
  filename: string;
  reason: string;
};

export type BatchStatus = {
  batchId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'partial' | 'cancelled';
  sourceFilename: string;
  acceptedFiles: BatchAcceptedFile[];
  rejectedFiles: BatchRejectedFile[];
  jobIds: string[];
  statusCounts: Record<string, number>;
  createdAt: string;
  updatedAt: string;
  persistenceWarning?: string | null;
};

export type ViolationEvent = {
  id: string;
  type: string;
  name: string;
  severity: Severity;
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

export type AnalysisResult = {
  jobId?: string;
  status?: 'completed';
  createdAt?: string | null;
  queuedAt?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  completedAt?: string | null;
  elapsedSeconds?: number | null;
  queueWaitSeconds?: number | null;
  analysisRuntimeSeconds?: number | null;
  id: string;
  query: string;
  media: MediaItem;
  summary?: {
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
  violations?: Array<{
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
  }>;
  framesAnalyzed: number;
  generatedAt: string;
  summaryText?: string;
  settings?: AnalysisSettings;
  events?: ViolationEvent[];
  frames: FrameResult[];
  technicalDetails?: Record<string, unknown> | null;
};
