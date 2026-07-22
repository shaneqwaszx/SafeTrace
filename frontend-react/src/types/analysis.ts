export type Severity = 'High' | 'Medium' | 'Low';

export type MediaType = 'video' | 'image' | 'unknown';

export type MediaStatus = 'draft' | 'ready' | 'queued' | 'processing' | 'completed' | 'error';

export type DeviceMode = 'Auto' | 'CPU' | 'GPU';

export type BackendConnectionState = 'live' | 'connecting' | 'connected' | 'disconnected' | 'incompatible' | 'error';

export type RecoveryJob = {
  jobId: string;
  batchId?: string | null;
  sourceRelativePath: string;
  sourceGroupPath: string;
  status: string;
  lastStage: string;
  progressPercent: number;
  resumable: boolean;
  resumableReason: string;
  retryCount: number;
  storedBytes: number;
  lastUpdate: string;
};

export type RecoverySummary = {
  policy: 'prompt' | 'auto_resume' | 'discard' | string;
  candidateCount: number;
  storedBytes: number;
  jobs: RecoveryJob[];
  batches: Array<{ batchId?: string | null; sourceFilename: string; candidateCount: number; storedBytes: number; jobs: RecoveryJob[] }>;
};

export type StorageSummary = {
  capturedAt: string;
  runtimeRoot: string;
  categories: Record<string, number>;
  totalManagedBytes: number;
  freeBytes: number;
  totalDiskBytes: number;
  minimumFreeBytes: number;
  storagePressure: boolean;
  retention: Record<string, number | string>;
  protected: Record<string, number>;
};

export type DashboardSummary = {
  capturedAt: string;
  cards: Record<string, number | null>;
  jobsByStatus: Record<string, number>;
  violationsByType: Record<string, number>;
  videosBySourceGroup: Record<string, number>;
  runtimeByMode: Record<string, number>;
  failuresByReason: Record<string, number>;
  storageByCategory: Record<string, number>;
  accuracyMetricsAvailable: boolean;
  accuracyNotice: string;
};

export type ExportSummary = {
  exportId: string;
  jobId?: string;
  batchId?: string | null;
  status: string;
  createdAt?: string;
  sizeBytes: number;
  fileCount: number;
  verified: boolean;
  path: string;
  manifestPath: string;
};

export type PaginatedResponse<T> = {
  page: number;
  pageSize: number;
  total: number;
  items: T[];
};

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
  reviewMode: 'fast_local' | 'comprehensive';
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
  originProfileId?: string | null;
  originProfileLabel?: string | null;
  originRule?: string | null;
  profileApplicability?: { applicable?: boolean; reason?: string; evaluatedAsProfileId?: string } | null;
  reviewLevel?: string | null;
  deduplicationKey?: string | null;
  findingId?: string;
  evidenceId?: string;
  jobId?: string;
  batchId?: string | null;
  sourceChecksum?: string | null;
  originalFilename?: string;
  sourceRelativePath?: string;
  sourceGroupPath?: string;
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
  sourceMetadata?: Record<string, unknown>;
  videoFilename?: string;
  sourceRelativePath?: string;
  sourceGroup?: string;
  batchId?: string | null;
  jobId?: string | null;
  findingId?: string | null;
  findingIds?: string[];
  evidenceId?: string;
  mediaArtifactId?: string | null;
  sourceChecksum?: string | null;
  originalFilename?: string;
  sourceGroupPath?: string;
  timestampSeconds?: number;
  timestampLabel?: string;
  sourceFrameIndex?: number | null;
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
  jobRuntimeLabel?: string | null;
  requestedModeLabel?: string | null;
  actualReviewLabel?: string | null;
  actualDeviceLabel?: string | null;
  vlmStatusLabel?: string | null;
  mobileSamStatusLabel?: string | null;
  elapsedSeconds?: number | null;
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
  reviewMode?: 'fast_local' | 'comprehensive';
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
  reviewMode?: 'fast_local' | 'comprehensive';
  importKey?: string;
};

export type AnalysisJob = {
  jobId: string;
  status: 'queued' | 'waiting_for_capacity' | 'waiting_for_job_slot' | 'waiting_for_gpu' | 'waiting_for_mobilesam' | 'waiting_for_vlm' | 'running_preprocess' | 'running_detector' | 'running_refinement' | 'running_report' | 'running' | 'retry_wait' | 'recovering' | 'paused' | 'completed' | 'failed' | 'cancelled';
};

export type AnalysisSetupSummary = {
  readOnly?: boolean;
  requestedCoverage?: { id?: string; label?: string };
  actualCoverage?: { id?: string; label?: string };
  profile?: { id?: string; label?: string };
  query?: string;
  childSettingsDiffer?: boolean;
  batchChildCount?: number;
  frameSampling?: {
    strategy?: string | null;
    requestedFps?: number | null;
    samplingFps?: number | null;
    samplingIntervalSeconds?: number | null;
    sampledFrameCount?: number | null;
    sourceVideoFrameCount?: number | null;
    processingWindowCount?: number | null;
    maximumSampledFrames?: number | null;
    requestedEvidenceFrames?: number | null;
    aggregateSampledFrameCount?: number | null;
    aggregateSourceVideoFrameCount?: number | null;
  };
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
  scheduler?: {
    enqueueSequence?: number | null;
    batchEnqueueSequence?: number | null;
    childSequence?: number | null;
    queuePosition?: number | null;
    workerSlot?: number | null;
    capacityReason?: string | null;
    schedulerPolicy?: string | null;
  } | null;
  sourceMetadata?: Record<string, unknown>;
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
  stageStartedAt?: string | null;
  stageElapsedSeconds?: number | null;
  heartbeatAt?: string | null;
  persistenceWarning?: string | null;
  manifestPersistenceWarning?: string | null;
  requestedModeLabel?: string | null;
  requestedVisualExplanationMode?: string | null;
  actualExplanationMode?: string | null;
  finalExplanationSource?: string | null;
  explanationOutcome?: string | null;
  explanationOutcomeLabel?: string | null;
  vlmAttempted?: boolean | null;
  lightweightVlmAttempted?: boolean | null;
  enhancedVlmAttempted?: boolean | null;
  vlmAccepted?: boolean | null;
  lightweightVlmAccepted?: boolean | null;
  enhancedVlmAccepted?: boolean | null;
  vlmFallbackReason?: string | null;
  vlmFallbackReasonLabel?: string | null;
  actualDeviceLabel?: string | null;
  jobRuntimeLabel?: string | null;
  engineRuntimeSummary?: {
    requestedMode?: string | null;
    actualReview?: string | null;
    device?: string | null;
    vlm?: string | null;
    mobileSam?: string | null;
    runtime?: string | null;
    requestedDevice?: string | null;
    actualDetectorDevice?: string | null;
    actualMobileSamDevice?: string | null;
    actualLightweightVlmDevice?: string | null;
    actualEnhancedVlmDevice?: string | null;
    cudaAvailableAtJobStart?: boolean | null;
    gpuName?: string | null;
  } | null;
  analysisSetup?: AnalysisSetupSummary | null;
};

export type BatchAcceptedFile = {
  originalFilename: string;
  filename: string;
  sizeBytes: number;
  mediaType: 'video';
  jobId: string;
  status: AnalysisJob['status'];
  error?: string | null;
  sourceRelativePath?: string | null;
  sourceDirectory?: string | null;
  sourceGroupPath?: string | null;
  sourceGroupLabel?: string | null;
  checksumSha256?: string | null;
  importedAt?: string | null;
  fileIdentity?: string | null;
  violationCount?: number;
  enqueueSequence?: number | null;
  batchEnqueueSequence?: number | null;
  childSequence?: number | null;
  queuePosition?: number | null;
  workerSlot?: number | null;
  capacityReason?: string | null;
  requestedModeLabel?: string | null;
  actualReview?: string | null;
  deviceLabel?: string | null;
  mobileSamStatus?: string | null;
};

export type BatchRejectedFile = {
  filename: string;
  reason: string;
  sourceRelativePath?: string | null;
  category?: string | null;
};

export type BatchHierarchyNode = {
  name: string;
  path: string;
  type: 'batch' | 'group';
  children: BatchHierarchyNode[];
  files: BatchAcceptedFile[];
};

export type BatchStatus = {
  batchId: string;
  status: 'importing' | 'queued' | 'waiting_for_capacity' | 'waiting_for_job_slot' | 'waiting_for_gpu' | 'waiting_for_mobilesam' | 'waiting_for_vlm' | 'running_preprocess' | 'running_detector' | 'running_refinement' | 'running_report' | 'running' | 'retry_wait' | 'paused' | 'recovering' | 'completed' | 'completed_with_failures' | 'failed' | 'partial' | 'cancelled';
  sourceFilename: string;
  batchDisplayLabel?: string | null;
  importKey?: string | null;
  acceptedFiles: BatchAcceptedFile[];
  rejectedFiles: BatchRejectedFile[];
  jobIds: string[];
  statusCounts: Record<string, number>;
  groupSummaries?: Array<Record<string, unknown>>;
  hierarchy?: BatchHierarchyNode;
  throughput?: Record<string, unknown>;
  paused?: boolean;
  createdAt: string;
  updatedAt: string;
  persistenceWarning?: string | null;
  analysisSetup?: AnalysisSetupSummary | null;
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
    evidenceId?: string;
    mediaArtifactId?: string | null;
    jobId?: string;
    batchId?: string | null;
    sourceChecksum?: string | null;
    originalFilename?: string;
    sourceRelativePath?: string;
    sourceGroupPath?: string;
  }>;
  eventId?: string;
  jobId?: string;
  batchId?: string | null;
  sourceChecksum?: string | null;
  originalFilename?: string;
  sourceRelativePath?: string;
  sourceGroupPath?: string;
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
  requestedModeLabel?: string | null;
  requestedVisualExplanationMode?: string | null;
  actualExplanationMode?: string | null;
  finalExplanationSource?: string | null;
  explanationOutcome?: string | null;
  explanationOutcomeLabel?: string | null;
  vlmAttempted?: boolean | null;
  lightweightVlmAttempted?: boolean | null;
  enhancedVlmAttempted?: boolean | null;
  vlmAccepted?: boolean | null;
  lightweightVlmAccepted?: boolean | null;
  enhancedVlmAccepted?: boolean | null;
  vlmFallbackReason?: string | null;
  vlmFallbackReasonLabel?: string | null;
  actualDeviceLabel?: string | null;
  jobRuntimeLabel?: string | null;
  engineRuntimeSummary?: JobStatus['engineRuntimeSummary'];
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
    violationsDetected?: boolean;
    acceptedFindingCount?: number;
    evidenceStatus?: string;
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
      evidenceId?: string;
      mediaArtifactId?: string | null;
      findingId?: string;
      jobId?: string;
      batchId?: string | null;
      sourceChecksum?: string | null;
      originalFilename?: string;
      sourceRelativePath?: string;
      sourceGroupPath?: string;
    }>;
    confidenceMin: number;
    confidenceMax: number;
    findingId?: string;
    jobId?: string;
    batchId?: string | null;
    sourceChecksum?: string | null;
    originalFilename?: string;
    sourceRelativePath?: string;
    sourceGroupPath?: string;
  }>;
  framesAnalyzed: number;
  generatedAt: string;
  summaryText?: string;
  settings?: AnalysisSettings;
  events?: ViolationEvent[];
  frames: FrameResult[];
  evidenceStatus?: 'available' | 'partial' | 'unavailable' | 'not_generated' | string;
  analysisSetup?: AnalysisSetupSummary | null;
  technicalDetails?: Record<string, unknown> | null;
  batchId?: string | null;
  sourceChecksum?: string | null;
  originalFilename?: string;
  sourceRelativePath?: string;
  sourceGroupPath?: string;
  executionIdentity?: Record<string, unknown>;
  resultSchemaVersion?: number;
};
