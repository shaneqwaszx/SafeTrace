"""Pydantic schemas for the local SafeTrace API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DeviceMode = Literal["auto", "cpu", "cuda"]
JobStatus = Literal[
    "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam",
    "waiting_for_vlm", "running_preprocess", "running_detector", "running_refinement", "running_report",
    "running", "retry_wait", "recovering", "paused",
    "completed", "failed", "cancelled",
]
BatchStatus = Literal[
    "importing", "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam",
    "waiting_for_vlm", "running_preprocess", "running_detector", "running_refinement", "running_report",
    "running", "retry_wait", "paused", "recovering",
    "completed", "completed_with_failures", "failed", "partial", "cancelled",
]
ChatAvailabilityState = Literal["available", "disabled", "missing_model", "missing_runtime", "loading", "unavailable"]
VlmProfileId = Literal["rule_based", "lightweight_256m", "lightweight_512m", "enhanced_2b", "enhanced_3b"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    api: Literal["safetrace-local"]
    version: str
    offline: bool


class ModelStatus(BaseModel):
    status: Literal["ready", "available", "missing", "missing_checkpoint", "missing_runtime", "disabled", "unavailable"]
    path: Optional[str] = None
    message: Optional[str] = None
    actionHint: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class VlmProfileStatus(BaseModel):
    id: VlmProfileId
    label: str
    installed: bool
    available: bool
    requiresActivation: bool
    resourceLevel: str
    path: Optional[str] = None
    message: Optional[str] = None
    statusCopy: Optional[str] = None
    deprecated: bool = False
    notViable: bool = False
    candidate: bool = False
    legacy: bool = False
    fallback: bool = False
    requiresGpu: bool = False
    deviceDecision: Optional[Dict[str, Any]] = None


class VlmSystemStatus(BaseModel):
    selectedProfile: VlmProfileId
    enabled: bool
    active: bool
    runtimeAvailable: bool
    profiles: List[VlmProfileStatus]
    message: str
    requestedVisualExplanationMode: Optional[str] = None
    actualExplanationMode: Optional[str] = None
    vlmAvailability: Optional[str] = None
    vlmSuppressedReason: Optional[str] = None
    fallbackReason: Optional[str] = None
    lightweightModelPathChecked: Optional[str] = None
    ruleBasedFallbackActive: Optional[bool] = None
    ruleBasedFallbackAvailable: Optional[bool] = True
    lightweightVlmWorkerEnabled: Optional[bool] = None
    lightweightVlmWorkerTimeoutSeconds: Optional[float] = None
    lightweightVlmExplanationSource: Optional[str] = None
    lightweightVlmEvidenceBudget: Optional[int] = None
    lightweightVlmFrameLimit: Optional[int] = None
    lightweightVlmJobTimeoutSeconds: Optional[float] = None
    lightweightVlmMaxQualityFailures: Optional[int] = None
    lightweightVlmPrimaryPolicy: Optional[str] = None
    lightweightVlmFallbackPolicy: Optional[str] = None
    lightweightVlmCpuPrefer256m: Optional[bool] = None
    enhancedVlmRequiresGpu: Optional[bool] = None
    deviceGateway: Optional[Dict[str, Any]] = None


class SystemStatusResponse(BaseModel):
    app_version: Optional[str] = None
    backend_version: Optional[str] = None
    build_mode: Optional[str] = None
    runtime_layout: Optional[str] = None
    safeMode: Optional[bool] = None
    device: str
    gpuAvailable: bool
    models: Dict[str, ModelStatus]
    limits: Optional[Dict[str, Any]] = None
    queue: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None
    preflight: Optional[Dict[str, Any]] = None
    vlm: Optional[VlmSystemStatus] = None


class VlmSettingsRequest(BaseModel):
    selectedProfile: VlmProfileId
    enabled: bool = False


class VlmSettingsResponse(VlmSystemStatus):
    pass


class ChatStatusResponse(BaseModel):
    enabled: bool
    available: bool
    state: ChatAvailabilityState
    status: ChatAvailabilityState
    enabled_mode: str
    provider: str
    model: Optional[str] = None
    model_path: Optional[str] = None
    model_exists: Optional[bool] = None
    runtime_available: Optional[bool] = None
    speed_profile: Optional[str] = None
    warmup_on_open: Optional[bool] = None
    fallback_available: Optional[bool] = None
    fallback_label: Optional[str] = None
    runtime_diagnostics: Optional[Dict[str, Any]] = None
    python_executable: Optional[str] = None
    expected_venv_python: Optional[str] = None
    running_in_expected_venv: Optional[bool] = None
    llama_cpp_import_status: Optional[str] = None
    llama_cpp_spec_found: Optional[bool] = None
    llama_cpp_import_error_type: Optional[str] = None
    llama_cpp_import_error_message: Optional[str] = None
    setup_command: Optional[str] = None
    restart_required: Optional[str] = None
    reason: Optional[str] = None
    action_hint: Optional[str] = None
    message: str


class ChatRequest(BaseModel):
    message: str
    job_id: Optional[str] = None
    batch_id: Optional[str] = None
    include_current_result: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    safeTraceOnly: bool
    modelProvider: str


class AnalyzeResponse(BaseModel):
    jobId: str
    status: JobStatus


class BatchAcceptedFile(BaseModel):
    originalFilename: str
    filename: str
    sizeBytes: int
    mediaType: Literal["video"]
    jobId: str
    status: JobStatus
    error: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceDirectory: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    sourceGroupLabel: Optional[str] = None
    checksumSha256: Optional[str] = None
    importedAt: Optional[str] = None
    fileIdentity: Optional[str] = None
    violationCount: int = 0
    enqueueSequence: Optional[int] = None
    batchEnqueueSequence: Optional[int] = None
    childSequence: Optional[int] = None
    queuePosition: Optional[int] = None
    workerSlot: Optional[int] = None
    capacityReason: Optional[str] = None
    requestedModeLabel: Optional[str] = None
    actualReview: Optional[str] = None
    deviceLabel: Optional[str] = None
    mobileSamStatus: Optional[str] = None


class SecondaryReviewRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    evidenceFrameIds: List[str] = Field(default_factory=list)
    profile: Optional[str] = None
    query: Optional[str] = None
    attempted: bool = True
    succeeded: bool = False
    accepted: bool = False
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    uncertainty: Optional[str] = None
    agreement: Optional[Literal["supports", "inconclusive", "disagrees"]] = None
    fallbackReason: Optional[str] = None
    latencySeconds: Optional[float] = None
    rawProviderMetadata: Optional[Dict[str, Any]] = None


class SecondaryReviewResponse(SecondaryReviewRequest):
    reviewId: str
    jobId: str
    authoritative: Literal[False] = False
    attachedAt: str
    error: Optional[str] = None


class BatchRejectedFile(BaseModel):
    filename: str
    reason: str
    sourceRelativePath: Optional[str] = None
    category: Optional[str] = None


class BatchResponse(BaseModel):
    batchId: str
    status: BatchStatus
    sourceFilename: str
    batchDisplayLabel: Optional[str] = None
    importKey: Optional[str] = None
    acceptedFiles: List[BatchAcceptedFile]
    rejectedFiles: List[BatchRejectedFile]
    jobIds: List[str]
    statusCounts: Dict[str, int]
    groupSummaries: List[Dict[str, Any]] = Field(default_factory=list)
    hierarchy: Dict[str, Any] = Field(default_factory=dict)
    throughput: Dict[str, Any] = Field(default_factory=dict)
    paused: bool = False
    createdAt: str
    updatedAt: str
    persistenceWarning: Optional[str] = None
    analysisSetup: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    progressPercent: int = Field(ge=0, le=100)
    stage: str
    message: str
    currentStep: str
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    componentDiagnostics: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None
    updatedAt: Optional[str] = None
    createdAt: Optional[str] = None
    queuedAt: Optional[str] = None
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    completedAt: Optional[str] = None
    failedAt: Optional[str] = None
    cancelledAt: Optional[str] = None
    elapsedSeconds: Optional[float] = None
    queueWaitSeconds: Optional[float] = None
    analysisRuntimeSeconds: Optional[float] = None
    heartbeatAt: Optional[str] = None
    persistenceWarning: Optional[str] = None
    manifestPersistenceWarning: Optional[str] = None
    requestedModeLabel: Optional[str] = None
    requestedVisualExplanationMode: Optional[str] = None
    actualExplanationMode: Optional[str] = None
    finalExplanationSource: Optional[str] = None
    explanationOutcome: Optional[str] = None
    explanationOutcomeLabel: Optional[str] = None
    vlmAttempted: Optional[bool] = None
    lightweightVlmAttempted: Optional[bool] = None
    enhancedVlmAttempted: Optional[bool] = None
    vlmAccepted: Optional[bool] = None
    lightweightVlmAccepted: Optional[bool] = None
    enhancedVlmAccepted: Optional[bool] = None
    vlmFallbackReason: Optional[str] = None
    vlmFallbackReasonLabel: Optional[str] = None
    actualDeviceLabel: Optional[str] = None
    jobRuntimeLabel: Optional[str] = None
    engineRuntimeSummary: Optional[Dict[str, Any]] = None
    sourceMetadata: Optional[Dict[str, Any]] = None
    retryCount: Optional[int] = None
    maxAttempts: Optional[int] = None
    nextRetryAt: Optional[str] = None
    recoveryAction: Optional[str] = None
    analysisSetup: Optional[Dict[str, Any]] = None


class MediaSummary(BaseModel):
    id: str
    name: str
    type: Literal["video", "image", "unknown"]
    sizeBytes: int
    durationSeconds: Optional[float] = None
    jobId: Optional[str] = None
    batchId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroupPath: Optional[str] = None


class AnalysisSummary(BaseModel):
    framesAnalyzed: int
    framesWithViolations: int
    uniqueViolationTypes: int
    highestSeverity: Optional[str] = None
    summaryText: str
    potentialEventCount: Optional[int] = None
    eventTypes: Optional[List[str]] = None
    overallConfidence: Optional[float] = None
    keyEvents: Optional[List[Dict[str, Any]]] = None
    jobId: Optional[str] = None
    batchId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    violationsDetected: Optional[bool] = None
    acceptedFindingCount: Optional[int] = None
    evidenceStatus: Optional[str] = None


class AffectedFrame(BaseModel):
    frameId: str
    frameNumber: int
    timestamp: str
    confidence: float
    evidenceId: Optional[str] = None
    mediaArtifactId: Optional[str] = None
    findingId: Optional[str] = None
    jobId: Optional[str] = None
    batchId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    evidenceStrength: Optional[str] = None
    confidenceReason: Optional[str] = None
    verifierAgreement: Optional[str] = None
    originProfileId: Optional[str] = None
    originProfileLabel: Optional[str] = None
    originRule: Optional[str] = None
    profileApplicability: Optional[Dict[str, Any]] = None
    reviewLevel: Optional[str] = None
    deduplicationKey: Optional[str] = None


class GroupedViolation(BaseModel):
    id: str
    name: str
    severity: str
    description: str
    affectedFrames: List[AffectedFrame]
    confidenceMin: float
    confidenceMax: float
    findingId: Optional[str] = None
    jobId: Optional[str] = None
    batchId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    evidenceStrength: Optional[str] = None
    confidenceReasons: Optional[List[str]] = None
    reviewRequired: Optional[bool] = None
    ruleSupport: Optional[str] = None
    verifierAgreement: Optional[str] = None
    verifierDisagreementReason: Optional[str] = None
    finalReviewerNote: Optional[str] = None
    originProfileId: Optional[str] = None
    originProfileLabel: Optional[str] = None
    originRule: Optional[str] = None
    profileApplicability: Optional[Dict[str, Any]] = None
    reviewLevel: Optional[str] = None
    deduplicationKey: Optional[str] = None


class FrameViolation(BaseModel):
    id: str
    name: str
    severity: str
    confidence: float
    description: str
    findingId: Optional[str] = None
    evidenceId: Optional[str] = None
    jobId: Optional[str] = None
    batchId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    evidenceStrength: Optional[str] = None
    confidenceReason: Optional[str] = None
    reviewRequired: Optional[bool] = None
    ruleSupport: Optional[str] = None
    suppressedFindings: Optional[List[Dict[str, Any]]] = None
    unsupportedRuleReason: Optional[str] = None
    verifierAgreement: Optional[str] = None
    verifierDisagreementReason: Optional[str] = None
    verifierConfidenceHint: Optional[str] = None
    finalReviewerNote: Optional[str] = None
    originProfileId: Optional[str] = None
    originProfileLabel: Optional[str] = None
    originRule: Optional[str] = None
    profileApplicability: Optional[Dict[str, Any]] = None
    reviewLevel: Optional[str] = None
    deduplicationKey: Optional[str] = None


class EventSupportingFrame(BaseModel):
    frameId: str
    frameNumber: int
    timestamp: str
    confidence: float
    imageUrl: Optional[str] = None
    batchId: Optional[str] = None
    jobId: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    timestampSeconds: Optional[float] = None
    timestampLabel: Optional[str] = None
    eventId: Optional[str] = None
    findingId: Optional[str] = None
    evidenceId: Optional[str] = None
    mediaArtifactId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    originProfileId: Optional[str] = None
    originProfileLabel: Optional[str] = None
    originRule: Optional[str] = None
    profileApplicability: Optional[Dict[str, Any]] = None
    reviewLevel: Optional[str] = None
    deduplicationKey: Optional[str] = None


class ViolationEvent(BaseModel):
    id: str
    type: str
    name: str
    severity: str
    description: str
    startTimestamp: str
    endTimestamp: str
    representativeConfidence: float
    confidenceMin: float
    confidenceMax: float
    supportingFrameCount: int
    supportingFrames: List[EventSupportingFrame]
    batchId: Optional[str] = None
    jobId: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    eventId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    originProfileId: Optional[str] = None
    originProfileLabel: Optional[str] = None
    originRule: Optional[str] = None
    profileApplicability: Optional[Dict[str, Any]] = None
    reviewLevel: Optional[str] = None
    deduplicationKey: Optional[str] = None


class FrameResult(BaseModel):
    id: str
    frameNumber: int
    timestamp: str
    queryRelevance: float
    status: Literal["violations_detected", "no_violations"]
    imageUrl: Optional[str] = None
    imageMessage: Optional[str] = None
    explanationSource: Optional[Literal["vlm", "vlm_local", "vlm_ollama", "vlm_lightweight", "vlm_enhanced", "rule_template_plus_vlm", "rule_template_plus_lightweight_vlm", "rule_template_plus_lightweight_plus_enhanced", "rule_based"]] = None
    violations: List[FrameViolation]
    technicalEvidence: Dict[str, Any]
    sourceMetadata: Optional[Dict[str, Any]] = None
    videoFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroup: Optional[str] = None
    batchId: Optional[str] = None
    jobId: Optional[str] = None
    findingId: Optional[str] = None
    findingIds: Optional[List[str]] = None
    evidenceId: Optional[str] = None
    mediaArtifactId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    timestampSeconds: Optional[float] = None
    timestampLabel: Optional[str] = None
    sourceFrameIndex: Optional[int] = None
    presentationNumber: Optional[int] = None
    sceneApplicability: Optional[Dict[str, Any]] = None
    suppressedFindings: Optional[List[Dict[str, Any]]] = None


class AnalysisResultResponse(BaseModel):
    jobId: str
    status: Literal["completed"]
    createdAt: Optional[str] = None
    queuedAt: Optional[str] = None
    startedAt: Optional[str] = None
    finishedAt: Optional[str] = None
    completedAt: Optional[str] = None
    elapsedSeconds: Optional[float] = None
    queueWaitSeconds: Optional[float] = None
    analysisRuntimeSeconds: Optional[float] = None
    media: MediaSummary
    query: str
    summary: AnalysisSummary
    violations: List[GroupedViolation]
    events: Optional[List[ViolationEvent]] = None
    frames: List[FrameResult]
    evidence: Optional[List[FrameResult]] = None
    evidenceStatus: Optional[str] = None
    diagnosticFrames: Optional[List[FrameResult]] = None
    engineMetrics: Optional[Dict[str, Any]] = None
    technicalDetails: Optional[Dict[str, Any]] = None
    sourceMetadata: Optional[Dict[str, Any]] = None
    reviewMode: Optional[str] = None
    requestedReviewMode: Optional[str] = None
    batchId: Optional[str] = None
    sourceChecksum: Optional[str] = None
    originalFilename: Optional[str] = None
    sourceRelativePath: Optional[str] = None
    sourceGroupPath: Optional[str] = None
    executionIdentity: Optional[Dict[str, Any]] = None
    resultSchemaVersion: Optional[int] = None
    analysisSetup: Optional[Dict[str, Any]] = None
