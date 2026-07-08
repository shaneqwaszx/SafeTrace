import { AlertTriangle, BarChart3, ClipboardCheck, Copy, Database, RefreshCcw, Server, Trash2, UploadCloud } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AnalysisProgress } from './components/AnalysisProgress';
import { AnalysisSummary } from './components/AnalysisSummary';
import { AnnotationViewer } from './components/AnnotationViewer';
import { AppShell } from './components/AppShell';
import { EvidenceFrames } from './components/EvidenceFrames';
import { MediaLibraryPanel } from './components/MediaLibraryPanel';
import { QueryTabs } from './components/QueryTabs';
import { ReportActions } from './components/ReportActions';
import { SafeTraceAssistant } from './components/SafeTraceAssistant';
import { SafetyInsightsDashboard } from './components/SafetyInsightsDashboard';
import { Sidebar } from './components/Sidebar';
import { StatisticsPanel } from './components/StatisticsPanel';
import { TimelineVisualization } from './components/TimelineVisualization';
import { UploadPanel } from './components/UploadPanel';
import { VideoQueue } from './components/VideoQueue';
import { ViolationSummary } from './components/ViolationSummary';
import { sampleMedia } from './data/mockAnalysis';
import {
  buildEffectiveProfileQuery,
  getProfileQueryConflict,
  getUseCaseProfileDefaultQuery,
  resolveUseCaseProfile,
} from './data/useCaseProfiles';
import {
  SAFETRACE_API_BASE,
  SAFETRACE_API_BASE_CANDIDATES,
  SAFETRACE_ENABLE_PREVIEW_MODE,
  SAFETRACE_REQUIRE_BACKEND,
  BackendDiscoveryError,
  checkBackendHealth,
  deleteJob,
  discoverBackendRuntime,
  getBatchStatus,
  getActiveApiBase,
  getJobResult,
  getJobStatus,
  getMockMediaLibrary,
  runBackendAnalysis,
  runBackendBatchAnalysis,
  runMockAnalysis,
  updateVlmSettings,
} from './services/analysisService';
import {
  type CachedResultEntry,
  clearCachedResults,
  clearSafeTraceResultCacheStorageKeys,
  deleteCachedResult,
  isCacheEntryStale,
  jobCacheKey,
  loadCachedResults,
  mediaCacheKey,
  saveCachedResult,
} from './services/resultCache';
import type {
  AnalysisResult,
  AnalysisSettings,
  BatchStatus,
  BackendConnectionState,
  JobStatus,
  MediaItem,
  SystemStatus,
  VlmExplanationProfileId,
} from './types/analysis';
import { formatFileSize } from './utils/formatters';
import { copyJobIdToClipboard, formatShortJobId } from './utils/jobIds';
import { SelectedMediaViewer } from './components/SelectedMediaViewer';

const DEFAULT_QUERY = resolveUseCaseProfile().defaultQuery;
const ANALYSIS_STEPS = [
  'Preparing selected media',
  'Sampling frames',
  'Matching query against visual evidence',
  'Grouping safety findings',
  'Preparing evidence report',
];
const SAMPLE_QUERY_BY_MEDIA_ID: Record<string, string> = {
  'media-2026-06-18-113842': 'worker without helmet',
  'media-sample-loading-bay': 'worker inside restricted loading bay',
  'media-sample-maintenance': 'worker without helmet',
};

const TERMINAL_JOB_STATUSES = new Set(['completed', 'failed', 'cancelled']);

function jobStatusToMediaStatus(status: JobStatus['status']): MediaItem['status'] {
  if (status === 'queued') return 'queued';
  if (status === 'running') return 'processing';
  if (status === 'completed') return 'completed';
  return 'error';
}

function isBatchId(value?: string | null): boolean {
  return typeof value === 'string' && value.startsWith('batch_');
}

function isJobId(value?: string | null): boolean {
  return typeof value === 'string' && value.startsWith('job_');
}

function isTerminalCachedJob(entry: CachedResultEntry): boolean {
  const status = entry.jobStatus?.status ?? entry.status;
  return Boolean(entry.result) || TERMINAL_JOB_STATUSES.has(status);
}

function collectTerminalCachedJobIds(entries: CachedResultEntry[]): string[] {
  const jobIds = new Set<string>();
  entries.forEach((entry) => {
    if (!isTerminalCachedJob(entry)) return;
    [entry.selectedJobId, entry.jobId, entry.result?.jobId].forEach((jobId) => {
      if (typeof jobId === 'string' && isJobId(jobId)) jobIds.add(jobId);
    });
  });
  return Array.from(jobIds);
}

function withEffectiveProfile(
  profile: AnalysisSettings['useCaseProfile'],
  requestedQuery: string,
): AnalysisSettings['useCaseProfile'] {
  const requested = requestedQuery.trim() || profile.defaultQuery;
  const effective = buildEffectiveProfileQuery(profile, requested);
  return {
    ...resolveUseCaseProfile(profile.profileId, profile.customText ?? '', requested),
    requestedQuery: requested,
    effectiveQuery: effective,
  };
}
const VLM_SELECTED_PROFILE_STORAGE_KEY = 'safetrace:vlm:selectedProfile';
const VLM_ENABLED_STORAGE_KEY = 'safetrace:vlm:enabled';
const VLM_PROFILE_IDS: VlmExplanationProfileId[] = [
  'rule_based',
  'lightweight_256m',
  'lightweight_512m',
  'enhanced_2b',
  'enhanced_3b',
];
const JOB_POLL_TIMEOUT_MS = 5 * 60 * 1000;
const BATCH_POLL_TIMEOUT_MS = 10 * 60 * 1000;
const FRESH_BACKEND_HEARTBEAT_MS = 45 * 1000;
const PERSIST_BROWSER_RESULT_CACHE = false;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function isVlmProfileId(value: string | null | undefined): value is VlmExplanationProfileId {
  return Boolean(value && VLM_PROFILE_IDS.includes(value as VlmExplanationProfileId));
}

function readLocalStorageValue(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeLocalStorageValue(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // The UI still works if localStorage is unavailable in this browser context.
  }
}

function getInitialVlmSettings(): Pick<AnalysisSettings, 'vlmProfile' | 'vlmEnabled'> {
  const storedProfile = readLocalStorageValue(VLM_SELECTED_PROFILE_STORAGE_KEY);
  const staleRemovedProfile = storedProfile === 'lightweight_256m'
    || storedProfile === 'enhanced_3b';
  const vlmProfile = isVlmProfileId(storedProfile) && !staleRemovedProfile ? storedProfile : 'rule_based';
  const vlmEnabled = readLocalStorageValue(VLM_ENABLED_STORAGE_KEY) === 'true' && vlmProfile !== 'rule_based';
  return { vlmProfile, vlmEnabled };
}

function isMainVlmSelectorProfileAvailable(status: SystemStatus | null, profile: VlmExplanationProfileId): boolean {
  if (profile === 'rule_based') return true;
  if (profile !== 'lightweight_512m' && profile !== 'enhanced_2b') return false;
  const backendProfile = status?.vlm?.profiles?.find((candidate) => candidate.id === profile);
  return Boolean(backendProfile?.installed && backendProfile?.available);
}

function shouldRequestVlm(settings: AnalysisSettings): boolean {
  return settings.visualExplanations && settings.vlmProfile !== 'rule_based' && settings.vlmEnabled;
}

function parseTimestampMs(value?: string | null): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function isBackendHeartbeatFresh(status: Pick<JobStatus, 'heartbeatAt' | 'updatedAt'>, now = Date.now()): boolean {
  const timestamp = parseTimestampMs(status.heartbeatAt || status.updatedAt);
  return timestamp !== null && now - timestamp <= FRESH_BACKEND_HEARTBEAT_MS;
}

function isBatchUpdateFresh(status: Pick<BatchStatus, 'updatedAt'>, now = Date.now()): boolean {
  const timestamp = parseTimestampMs(status.updatedAt);
  return timestamp !== null && now - timestamp <= FRESH_BACKEND_HEARTBEAT_MS;
}

function hasUsableCachedPayload(
  entry: CachedResultEntry,
  entriesByKey: Record<string, CachedResultEntry>,
): boolean {
  if (isCacheEntryStale(entry)) return false;
  if (entry.result) return true;
  if (entry.selectedJobId && entriesByKey[jobCacheKey(entry.selectedJobId)]?.result) return true;
  return false;
}

class BackendJobFailureError extends Error {
  debugDetails: string;

  constructor(message: string, debugDetails: string) {
    super(message);
    this.name = 'BackendJobFailureError';
    this.debugDetails = debugDetails;
  }
}

function metricString(metrics: JobStatus['metrics'] | undefined, key: string): string | null {
  const value = metrics?.[key];
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function jobFailureDebugDetails(status: JobStatus): string {
  return (
    metricString(status.metrics, 'errorType')
    || metricString(status.metrics, 'errorMessage')
    || status.error
    || status.currentStep
    || 'Backend job failed without additional diagnostics.'
  );
}

function resultComponentDiagnostics(result: AnalysisResult | null): Record<string, unknown> | null {
  if (!result?.technicalDetails || typeof result.technicalDetails !== 'object') return null;
  const technicalDetails = result.technicalDetails as Record<string, unknown>;
  const direct = technicalDetails.componentDiagnostics;
  if (direct && typeof direct === 'object') return direct as Record<string, unknown>;
  const jobMetrics = technicalDetails.jobMetrics;
  if (jobMetrics && typeof jobMetrics === 'object') {
    const nested = (jobMetrics as Record<string, unknown>).componentDiagnostics;
    if (nested && typeof nested === 'object') return nested as Record<string, unknown>;
  }
  return null;
}

function persistVlmSettings(settings: AnalysisSettings): void {
  writeLocalStorageValue(VLM_SELECTED_PROFILE_STORAGE_KEY, settings.vlmProfile);
  writeLocalStorageValue(VLM_ENABLED_STORAGE_KEY, String(settings.vlmProfile !== 'rule_based' && settings.vlmEnabled));
}

function isZipFile(file: File): boolean {
  return file.name.toLowerCase().endsWith('.zip') || file.type === 'application/zip';
}

function getMediaTypeForFiles(files: File[]): MediaItem['type'] {
  if (files.length !== 1) return 'unknown';
  const file = files[0];
  if (isZipFile(file)) return 'unknown';
  if (file.type.startsWith('image/')) return 'image';
  return 'video';
}

function getSelectionName(files: File[]): string {
  if (files.length === 1) return files[0].name;
  return `${files.length} selected videos`;
}

function getSelectionSize(files: File[]): string {
  const total = files.reduce((sum, file) => sum + file.size, 0);
  return formatFileSize(total);
}

function App() {
  const previewMode = SAFETRACE_ENABLE_PREVIEW_MODE;
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [mediaLibrary, setMediaLibrary] = useState<MediaItem[]>(previewMode ? [sampleMedia] : []);
  const [selectedMedia, setSelectedMedia] = useState<MediaItem | null>(previewMode ? sampleMedia : null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [settings, setSettings] = useState<AnalysisSettings>(() => {
    const vlmSettings = getInitialVlmSettings();
    return {
      fps: 1,
      topK: 5,
      visualExplanations: true,
      vlmProfile: vlmSettings.vlmProfile,
      vlmEnabled: vlmSettings.vlmEnabled,
      enhancedVlmExplanations: vlmSettings.vlmEnabled,
      deviceMode: 'Auto',
      useCaseProfile: resolveUseCaseProfile(),
    };
  });
  const [activeAnalysisMediaIds, setActiveAnalysisMediaIds] = useState<Record<string, true>>({});
  const [activeStep, setActiveStep] = useState(0);
  const [highlightedFrameId, setHighlightedFrameId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [showAnnotation, setShowAnnotation] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [hoveredFrameId, setHoveredFrameId] = useState<string | null>(null);
  const [backendState, setBackendState] = useState<BackendConnectionState>('live');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [backendMessage, setBackendMessage] = useState<string | null>(null);
  const [activeApiBase, setActiveApiBase] = useState(SAFETRACE_API_BASE);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const [selectedBatchJobId, setSelectedBatchJobId] = useState<string | null>(null);
  const [batchResultLoadingJobId, setBatchResultLoadingJobId] = useState<string | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'backend' | 'preview' | null>(null);
  const [cachedEntries, setCachedEntries] = useState<Record<string, CachedResultEntry>>({});
  const [cacheMessage, setCacheMessage] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'analysis' | 'insights'>('analysis');
  const localFilesRef = useRef<Record<string, File>>({});
  const localFileGroupsRef = useRef<Record<string, File[]>>({});
  const objectUrlsRef = useRef<Set<string>>(new Set());
  const selectedMediaIdRef = useRef<string | null>(selectedMedia?.id ?? null);
  const activeAnalysisMediaIdsRef = useRef<Record<string, true>>({});

  const backendConnected = backendState === 'connected';
  const controlsLocked = SAFETRACE_REQUIRE_BACKEND && !backendConnected && !previewMode;
  const canUsePreview = previewMode && Boolean(selectedMedia);
  const cachedEntryCount = Object.keys(cachedEntries).length;
  const hasAnyActiveAnalysis = Object.keys(activeAnalysisMediaIds).length > 0;
  const isLoading = Boolean(selectedMedia && activeAnalysisMediaIds[selectedMedia.id]);

  useEffect(() => {
    selectedMediaIdRef.current = selectedMedia?.id ?? null;
  }, [selectedMedia?.id]);

  useEffect(() => {
    activeAnalysisMediaIdsRef.current = activeAnalysisMediaIds;
  }, [activeAnalysisMediaIds]);

  useEffect(() => {
    let isMounted = true;
    if (!PERSIST_BROWSER_RESULT_CACHE) {
      clearSafeTraceResultCacheStorageKeys();
      clearCachedResults()
        .catch(() => undefined)
        .finally(() => {
          if (isMounted) setCachedEntries({});
        });
      return () => {
        isMounted = false;
      };
    }
    loadCachedResults()
      .then((entries) => {
        if (!isMounted) return;
        const entriesByKey = Object.fromEntries(entries.map((entry) => [entry.cacheKey, entry]));
        const usableEntries = entries.filter((entry) => hasUsableCachedPayload(entry, entriesByKey));
        const rejectedEntries = entries.filter((entry) => !hasUsableCachedPayload(entry, entriesByKey));
        const next = Object.fromEntries(usableEntries.map((entry) => [entry.cacheKey, entry]));
        setCachedEntries(next);
        if (rejectedEntries.length) {
          setCacheMessage(
            `${rejectedEntries.length} stale or incomplete browser cache item${rejectedEntries.length === 1 ? '' : 's'} ignored.`,
          );
          rejectedEntries.forEach((entry) => {
            void deleteCachedResult(entry.cacheKey).catch(() => undefined);
          });
        }
      })
      .catch(() => {
        if (isMounted) {
          setCacheMessage('Local result cache could not be opened in this browser session.');
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  function rememberCachedEntry(entry: CachedResultEntry) {
    setCachedEntries((current) => ({ ...current, [entry.cacheKey]: entry }));
    if (!PERSIST_BROWSER_RESULT_CACHE) return;
    void saveCachedResult(entry).then((result) => {
      if (!result.saved && result.reason) {
        setCacheMessage(result.reason);
      }
    }).catch(() => {
      setCacheMessage('Result is available in this tab, but could not be saved to the local browser cache.');
    });
  }

  function updateMediaStatus(mediaId: string, status: MediaItem['status']) {
    setMediaLibrary((current) => current.map((item) => (
      item.id === mediaId ? { ...item, status } : item
    )));
    setSelectedMedia((current) => (
      current?.id === mediaId ? { ...current, status } : current
    ));
  }

  function updateMediaItem(mediaId: string, patch: Partial<MediaItem>) {
    setMediaLibrary((current) => current.map((item) => (
      item.id === mediaId ? { ...item, ...patch } : item
    )));
    setSelectedMedia((current) => (
      current?.id === mediaId ? { ...current, ...patch } : current
    ));
  }

  function setMediaAnalysisActive(mediaId: string, active: boolean) {
    setActiveAnalysisMediaIds((current) => {
      if (active) return { ...current, [mediaId]: true };
      const next = { ...current };
      delete next[mediaId];
      return next;
    });
  }

  function updateMediaJobReference(
    mediaId: string,
    refs: Partial<Pick<MediaItem, 'jobId' | 'selectedJobId' | 'batchId'>>,
  ) {
    setMediaLibrary((current) => current.map((item) => (
      item.id === mediaId ? { ...item, ...refs } : item
    )));
    setSelectedMedia((current) => (
      current?.id === mediaId ? { ...current, ...refs } : current
    ));
  }

  function buildCacheEntry({
    media,
    result,
    jobStatus: nextJobStatus,
    batchStatus: nextBatchStatus,
    selectedJobId,
    source = 'backend',
    status,
    queryText,
  }: {
    media: MediaItem;
    result?: AnalysisResult;
    jobStatus?: JobStatus | null;
    batchStatus?: BatchStatus | null;
    selectedJobId?: string | null;
    source?: CachedResultEntry['source'];
    status?: string;
    queryText?: string;
  }): CachedResultEntry {
    const existing = cachedEntries[mediaCacheKey(media.id)];
    const candidateJobId = result?.jobId ?? nextJobStatus?.jobId ?? existing?.jobId;
    const safeJobId = isJobId(candidateJobId) ? candidateJobId : undefined;
    const candidateSelectedJobId = selectedJobId ?? existing?.selectedJobId;
    const safeSelectedJobId = isJobId(candidateSelectedJobId) ? candidateSelectedJobId : undefined;
    const staleBatchId = isBatchId(existing?.jobId) ? existing?.jobId : undefined;
    return {
      cacheKey: mediaCacheKey(media.id),
      cacheVersion: 1,
      mediaId: media.id,
      mediaName: media.filename,
      query: result?.query ?? queryText ?? media.effectiveQuery ?? media.requestedQuery ?? query,
      source,
      status: status ?? result?.status ?? nextBatchStatus?.status ?? nextJobStatus?.status ?? media.status,
      savedAt: existing?.savedAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      jobId: safeJobId,
      batchId: nextBatchStatus?.batchId ?? existing?.batchId ?? staleBatchId,
      selectedJobId: safeSelectedJobId,
      result: result ?? existing?.result,
      jobStatus: nextJobStatus ?? existing?.jobStatus ?? null,
      batchStatus: nextBatchStatus ?? existing?.batchStatus ?? null,
    };
  }

  function restoreCachedMedia(media: MediaItem): boolean {
    const entry = cachedEntries[mediaCacheKey(media.id)];
    const childEntry = entry?.selectedJobId ? cachedEntries[jobCacheKey(entry.selectedJobId)] : undefined;
    const restoredResult = childEntry?.result ?? entry?.result;
    if (!entry && !childEntry) return false;
    if (entry && !hasUsableCachedPayload(entry, cachedEntries) && !restoredResult) {
      setCacheMessage('Ignored an incomplete browser cache reference for this media. Reconnect the local runtime or rerun analysis.');
      return false;
    }

    setAnalysisResult(restoredResult ?? null);
    setJobStatus(childEntry?.jobStatus ?? entry?.jobStatus ?? null);
    setBatchStatus(entry?.batchStatus ?? null);
    setSelectedBatchJobId(entry?.selectedJobId ?? childEntry?.jobId ?? null);
    setAnalysisMode(restoredResult ? entry?.source ?? childEntry?.source ?? 'backend' : null);
    if (entry && isCacheEntryStale(entry)) {
      setCacheMessage('Showing an older cached result. Reconnect the local runtime to refresh it, or clear the cache.');
    }
    return true;
  }

  function handleOpenDashboardEntry(entry: CachedResultEntry) {
    if (!entry.result) return;
    setError(null);
    setErrorDetails(null);
    setCacheMessage(null);
    setAnalysisResult(entry.result);
    setJobStatus(entry.jobStatus ?? null);
    setBatchStatus(entry.batchStatus ?? null);
    setSelectedBatchJobId(entry.selectedJobId ?? entry.jobId ?? null);
    setAnalysisMode(entry.source);
    setActiveStep(ANALYSIS_STEPS.length);
    setActiveView('analysis');
    const knownMedia = mediaLibrary.find((media) => media.id === entry.mediaId);
    if (knownMedia) {
      setSelectedMedia({
        ...knownMedia,
        jobId: entry.jobId ?? entry.result.jobId ?? knownMedia.jobId,
        selectedJobId: entry.selectedJobId ?? knownMedia.selectedJobId,
        batchId: entry.batchId ?? knownMedia.batchId,
        requestedQuery: entry.result.media.requestedQuery ?? entry.query,
        effectiveQuery: entry.result.query,
        useCaseProfile: entry.result.settings?.useCaseProfile ?? entry.result.media.useCaseProfile ?? knownMedia.useCaseProfile,
      });
      if (entry.result.settings?.useCaseProfile ?? entry.result.media.useCaseProfile) {
        setSettings((current) => ({
          ...current,
          useCaseProfile: entry.result?.settings?.useCaseProfile ?? entry.result?.media.useCaseProfile ?? current.useCaseProfile,
        }));
      }
      setQuery(entry.result.query);
      return;
    }
    setSelectedMedia({
      id: entry.mediaId,
      filename: entry.mediaName,
      type: entry.result.media.type,
      sizeLabel: entry.result.media.sizeLabel,
      uploadedAt: entry.savedAt,
      status: entry.status === 'completed' ? 'completed' : 'ready',
      source: 'local',
      jobId: entry.jobId ?? entry.result.jobId,
      selectedJobId: entry.selectedJobId,
      batchId: entry.batchId,
      requestedQuery: entry.result.media.requestedQuery ?? entry.query,
      effectiveQuery: entry.result.query,
      useCaseProfile: entry.result.settings?.useCaseProfile ?? entry.result.media.useCaseProfile,
    });
    if (entry.result.settings?.useCaseProfile ?? entry.result.media.useCaseProfile) {
      setSettings((current) => ({
        ...current,
        useCaseProfile: entry.result?.settings?.useCaseProfile ?? entry.result?.media.useCaseProfile ?? current.useCaseProfile,
      }));
    }
    setQuery(entry.result.query);
  }

  async function refreshCachedResultFromBackend(media: MediaItem) {
    const entry = cachedEntries[mediaCacheKey(media.id)];
    const staleBatchId = isBatchId(entry?.jobId) ? entry?.jobId : undefined;
    const batchId = entry?.batchId ?? staleBatchId;
    const jobId = [entry?.selectedJobId, entry?.jobId].find((candidate) => isJobId(candidate));
    if (!backendConnected) return;
    try {
      if (batchId && !jobId) {
        const batch = await getBatchStatus(batchId);
        rememberCachedEntry(buildCacheEntry({
          media,
          batchStatus: batch,
          status: batch.status,
          queryText: media.effectiveQuery ?? media.requestedQuery,
        }));
        updateMediaJobReference(media.id, { batchId, jobId: undefined, selectedJobId: undefined });
        if (selectedMediaIdRef.current === media.id) setBatchStatus(batch);
        return;
      }
      if (!jobId) return;
      const status = await getJobStatus(jobId);
      const nextEntry = buildCacheEntry({ media, jobStatus: status, status: status.status });
      rememberCachedEntry(nextEntry);
      updateMediaJobReference(media.id, { jobId, selectedJobId: jobId });
      if (selectedMediaIdRef.current === media.id) setJobStatus(status);
      if (status.status !== 'completed') return;

      const result = await getJobResult(jobId);
      const resultProfile = result.settings?.useCaseProfile ?? result.media.useCaseProfile ?? media.useCaseProfile;
      result.settings = { ...settings, useCaseProfile: resultProfile ?? settings.useCaseProfile };
      result.media.useCaseProfile = resultProfile;
      result.media.requestedQuery = media.requestedQuery ?? result.query;
      result.media.effectiveQuery = result.query;
      const jobEntry: CachedResultEntry = {
        cacheKey: jobCacheKey(jobId),
        cacheVersion: 1,
        mediaId: media.id,
        mediaName: media.filename,
        query: result.query,
        source: 'backend',
        status: 'completed',
        savedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        jobId,
        batchId: entry?.batchId,
        result,
        jobStatus: status,
      };
      rememberCachedEntry(jobEntry);
      rememberCachedEntry(buildCacheEntry({
        media: {
          ...media,
          useCaseProfile: resultProfile,
          requestedQuery: media.requestedQuery ?? result.query,
          effectiveQuery: result.query,
        },
        result,
        jobStatus: status,
        selectedJobId: jobId,
        status: 'completed',
        queryText: result.query,
      }));
      if (selectedMediaIdRef.current === media.id) {
        setAnalysisResult(result);
        setAnalysisMode('backend');
      }
    } catch {
      if (entry?.result) {
        setCacheMessage('Showing cached result; the local runtime could not refresh it right now.');
      }
    }
  }

  const refreshBackendConnection = useCallback(async () => {
    setBackendState('connecting');
    setBackendMessage(null);
    try {
      const runtime = await discoverBackendRuntime();
      setActiveApiBase(runtime.apiBase);
      setSystemStatus(runtime.systemStatus);
      setBackendState('connected');
      setBackendMessage(null);
    } catch (err) {
      setActiveApiBase(getActiveApiBase());
      setSystemStatus(null);
      setBackendState(err instanceof BackendDiscoveryError ? err.state : 'disconnected');
      setBackendMessage(
        err instanceof Error
          ? err.message
          : 'SafeTrace Local Runtime is not reachable from this browser.',
      );
      if (!previewMode) {
        setMediaLibrary([]);
        setSelectedMedia(null);
        setSelectedFile(null);
        setSelectedFiles([]);
        setAnalysisResult(null);
        setBatchStatus(null);
        setSelectedBatchJobId(null);
      }
    }
  }, [previewMode]);

  useEffect(() => {
    void refreshBackendConnection();
  }, [refreshBackendConnection]);

  useEffect(() => {
    if (!backendConnected) return;
    if (!isMainVlmSelectorProfileAvailable(systemStatus, settings.vlmProfile)) {
      setSettings((current) => ({
        ...current,
        vlmProfile: 'rule_based',
        vlmEnabled: false,
      }));
      return;
    }
    void updateVlmSettings({
      selectedProfile: settings.vlmProfile,
      enabled: shouldRequestVlm(settings),
    }).catch(() => {
      // Older local runtimes do not expose VLM settings yet; local UI state remains authoritative.
    });
  }, [backendConnected, settings.visualExplanations, settings.vlmEnabled, settings.vlmProfile, systemStatus]);

  useEffect(() => {
    if (!previewMode) return undefined;
    let isMounted = true;
    getMockMediaLibrary()
      .then((items) => {
        if (isMounted) setMediaLibrary(items);
      })
      .catch(() => {
        if (isMounted) setMediaLibrary([sampleMedia]);
      });
    return () => { isMounted = false; };
  }, [previewMode]);

  useEffect(() => {
    return () => {
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      objectUrlsRef.current.clear();
    };
  }, []);

  function trackObjectUrl(url?: string) {
    if (url) objectUrlsRef.current.add(url);
  }

  function handleSelectMedia(media: MediaItem) {
    const knownSampleQueries = Object.values(SAMPLE_QUERY_BY_MEDIA_ID);
    const suggestedQuery = SAMPLE_QUERY_BY_MEDIA_ID[media.id];
    setSelectedMedia(media);
    setActiveView('analysis');
    const fileGroup = localFileGroupsRef.current[media.id] ?? (
      localFilesRef.current[media.id] ? [localFilesRef.current[media.id]] : []
    );
    setSelectedFiles(fileGroup);
    setSelectedFile(fileGroup.length === 1 ? fileGroup[0] : null);
    setError(null);
    setErrorDetails(null);
    setCacheMessage(null);
    if (media.useCaseProfile) {
      setSettings((current) => ({
        ...current,
        useCaseProfile: media.useCaseProfile ?? current.useCaseProfile,
      }));
      setQuery(media.requestedQuery ?? media.effectiveQuery ?? media.useCaseProfile.defaultQuery);
    }
    const restored = restoreCachedMedia(media);
    if (!restored) {
      setAnalysisResult(null);
      setJobStatus(null);
      setBatchStatus(null);
      setSelectedBatchJobId(null);
      setAnalysisMode(null);
    }
    void refreshCachedResultFromBackend(media);
    if (!media.useCaseProfile && suggestedQuery && knownSampleQueries.includes(query)) {
      setQuery(suggestedQuery);
    }
  }

  function handleDeleteMedia(mediaId: string) {
    const entry = cachedEntries[mediaCacheKey(mediaId)];
    const mediaToDelete = mediaLibrary.find((media) => media.id === mediaId);
    if (mediaToDelete?.previewUrl) {
      const stillUsedElsewhere = mediaLibrary.some((media) => media.id !== mediaId && media.previewUrl === mediaToDelete.previewUrl);
      if (!stillUsedElsewhere) {
        URL.revokeObjectURL(mediaToDelete.previewUrl);
        objectUrlsRef.current.delete(mediaToDelete.previewUrl);
      }
    }
    delete localFilesRef.current[mediaId];
    delete localFileGroupsRef.current[mediaId];
    setCachedEntries((current) => {
      const next = { ...current };
      delete next[mediaCacheKey(mediaId)];
      if (entry?.selectedJobId) delete next[jobCacheKey(entry.selectedJobId)];
      return next;
    });
    void deleteCachedResult(mediaCacheKey(mediaId));
    if (entry?.selectedJobId) void deleteCachedResult(jobCacheKey(entry.selectedJobId));
    setMediaLibrary((prev) => prev.filter((m) => m.id !== mediaId));
    if (selectedMedia?.id === mediaId && mediaLibrary.length > 1) {
      const next = mediaLibrary.find((m) => m.id !== mediaId);
      if (next) {
        setSelectedMedia(next);
        const nextFiles = localFileGroupsRef.current[next.id] ?? (
          localFilesRef.current[next.id] ? [localFilesRef.current[next.id]] : []
        );
        setSelectedFiles(nextFiles);
        setSelectedFile(nextFiles.length === 1 ? nextFiles[0] : null);
      }
    } else if (selectedMedia?.id === mediaId) {
      setSelectedMedia(null);
      setSelectedFile(null);
      setSelectedFiles([]);
    }
  }

  function handleSettingsChange(nextSettings: AnalysisSettings) {
    const profileChanged = nextSettings.useCaseProfile.profileId !== settings.useCaseProfile.profileId;
    const nextQuery = profileChanged ? getUseCaseProfileDefaultQuery(nextSettings.useCaseProfile) : query;
    const nextProfile = withEffectiveProfile(nextSettings.useCaseProfile, nextQuery);
    const normalizedSettings = {
      ...nextSettings,
      useCaseProfile: nextProfile,
      vlmEnabled: nextSettings.vlmProfile === 'rule_based' ? false : nextSettings.vlmEnabled,
      enhancedVlmExplanations: shouldRequestVlm(nextSettings),
    };
    persistVlmSettings(normalizedSettings);
    setSettings(normalizedSettings);
    if (profileChanged) setQuery(nextQuery);
    if (selectedMedia && !['queued', 'processing', 'completed'].includes(selectedMedia.status)) {
      updateMediaItem(selectedMedia.id, {
        useCaseProfile: nextProfile,
        requestedQuery: nextProfile.requestedQuery,
        effectiveQuery: nextProfile.effectiveQuery,
      });
    }
  }

  function handleFilesSelected(files: File[]) {
    if (controlsLocked) return;
    const selected = files.filter(Boolean);
    if (!selected.length) return;
    const hasActiveAnalysis = Object.keys(activeAnalysisMediaIdsRef.current).length > 0;
    const profileForDraft = withEffectiveProfile(settings.useCaseProfile, query);
    const isSinglePreviewableFile = selected.length === 1 && !isZipFile(selected[0]);
    const previewUrl = isSinglePreviewableFile ? URL.createObjectURL(selected[0]) : undefined;
    trackObjectUrl(previewUrl);
    const id = selected.length === 1
      ? `local-${selected[0].name}-${selected[0].lastModified}`
      : `local-batch-${Date.now()}`;
    const media: MediaItem = {
      id,
      filename: getSelectionName(selected),
      type: getMediaTypeForFiles(selected),
      sizeLabel: getSelectionSize(selected),
      uploadedAt: new Date().toISOString(),
      status: 'draft',
      source: 'local',
      previewUrl,
      useCaseProfile: profileForDraft,
      requestedQuery: profileForDraft.requestedQuery,
      effectiveQuery: profileForDraft.effectiveQuery,
    };
    if (selected.length === 1) {
      localFilesRef.current[media.id] = selected[0];
    }
    localFileGroupsRef.current[media.id] = selected;
    setMediaLibrary((prev) => [media, ...prev.filter((item) => item.id !== media.id)]);
    if (hasActiveAnalysis) {
      setCacheMessage(
        `${media.filename} was added as a draft job. The current analysis keeps running; select this item and click Send to queue it with the backend worker.`,
      );
      return;
    }
    setSelectedFile(selected.length === 1 ? selected[0] : null);
    setSelectedFiles(selected);
    setSelectedMedia(media);
    setAnalysisResult(null);
    setError(null);
    setErrorDetails(null);
    setJobStatus(null);
    setBatchStatus(null);
    setSelectedBatchJobId(null);
    setAnalysisMode(null);
  }

  function handleFileSelected(file: File) {
    handleFilesSelected([file]);
  }

  function progressToStep(progress: number) {
    if (progress >= 1) return ANALYSIS_STEPS.length;
    return Math.min(ANALYSIS_STEPS.length - 1, Math.max(0, Math.floor(progress * ANALYSIS_STEPS.length)));
  }

  function isBatchSelection(files: File[]) {
    return files.length > 1 || (files.length === 1 && isZipFile(files[0]));
  }

  async function pollBackendJob(jobId: string, media?: MediaItem): Promise<JobStatus> {
    const startedAt = Date.now();

    while (true) {
      let status: JobStatus;
      try {
        status = await getJobStatus(jobId);
      } catch (err) {
        try {
          await checkBackendHealth();
        } catch {
          await refreshBackendConnection();
          throw new BackendJobFailureError(
            'SafeTrace Local Runtime disconnected during analysis. Reconnect and retry this analysis.',
            err instanceof Error ? err.message : 'Job status could not be refreshed.',
          );
        }
        throw new BackendJobFailureError(
          'Analysis failed.',
          err instanceof Error ? err.message : 'Job status could not be refreshed while the backend was still healthy.',
        );
      }
      if (!media || selectedMediaIdRef.current === media.id) {
        setJobStatus(status);
        setActiveStep(progressToStep(status.progress));
      }
      if (media) {
        updateMediaStatus(media.id, jobStatusToMediaStatus(status.status));
        rememberCachedEntry(buildCacheEntry({
          media,
          jobStatus: status,
          status: status.status,
          queryText: media.effectiveQuery ?? media.requestedQuery,
        }));
      }

      if (status.status === 'completed') return status;
      if (status.status === 'failed' || status.status === 'cancelled') {
        throw new BackendJobFailureError(
          status.status === 'cancelled' ? 'Analysis cancelled.' : 'Analysis failed.',
          jobFailureDebugDetails(status),
        );
      }
      if (Date.now() - startedAt >= JOB_POLL_TIMEOUT_MS && !isBackendHeartbeatFresh(status)) {
        throw new Error('Analysis timed out because the backend heartbeat became stale. Check the local runtime logs, then retry.');
      }

      await wait(1200);
    }
  }

  async function pollBackendBatch(batchId: string, media?: MediaItem): Promise<BatchStatus> {
    const startedAt = Date.now();

    while (true) {
      let status: BatchStatus;
      try {
        status = await getBatchStatus(batchId);
      } catch (err) {
        try {
          await checkBackendHealth();
        } catch {
          await refreshBackendConnection();
          throw new BackendJobFailureError(
            'SafeTrace Local Runtime disconnected during batch analysis. Reconnect and retry this batch.',
            err instanceof Error ? err.message : 'Batch status could not be refreshed.',
          );
        }
        throw new BackendJobFailureError(
          'Batch analysis failed.',
          err instanceof Error ? err.message : 'Batch status could not be refreshed while the backend was still healthy.',
        );
      }
      if (!media || selectedMediaIdRef.current === media.id) {
        setBatchStatus(status);
      }
      const totalJobs = Math.max(status.acceptedFiles.length, 1);
      const completedJobs = status.acceptedFiles.filter((file) => (
        file.status === 'completed' || file.status === 'failed' || file.status === 'cancelled'
      )).length;
      const representativeJobIdCandidate = (
        status.acceptedFiles.find((file) => file.status === 'running')?.jobId
        ?? status.acceptedFiles.find((file) => file.status === 'queued')?.jobId
        ?? status.acceptedFiles.find((file) => file.status === 'completed')?.jobId
        ?? media?.selectedJobId
        ?? media?.jobId
        ?? ''
      );
      const representativeJobId = isJobId(representativeJobIdCandidate) ? representativeJobIdCandidate : '';
      const progressStatus: JobStatus = {
        jobId: representativeJobId,
        status: status.status === 'running' ? 'running' : status.status === 'queued' ? 'queued' : 'completed',
        progress: completedJobs / totalJobs,
        progressPercent: Math.round((completedJobs / totalJobs) * 100),
        stage: status.status === 'running' ? 'analyzing' : status.status,
        message: `Batch analysis ${status.status}`,
        currentStep: `Batch analysis ${status.status}`,
        error: status.status === 'failed' ? 'No videos completed successfully.' : null,
      };
      if (!media || selectedMediaIdRef.current === media.id) {
        setActiveStep(progressToStep(completedJobs / totalJobs));
        setJobStatus(progressStatus);
      }
      if (media) {
        updateMediaStatus(media.id, status.status === 'queued' ? 'queued' : status.status === 'running' ? 'processing' : 'completed');
        rememberCachedEntry(buildCacheEntry({
          media,
          batchStatus: status,
          status: status.status,
          queryText: media.effectiveQuery ?? media.requestedQuery,
        }));
      }

      if (status.status === 'completed' || status.status === 'partial') return status;
      if (status.status === 'failed' || status.status === 'cancelled') {
        throw new Error('Batch analysis could not be completed.');
      }
      if (Date.now() - startedAt >= BATCH_POLL_TIMEOUT_MS && !isBatchUpdateFresh(status)) {
        throw new Error('Batch analysis timed out because the backend status stopped updating. Check the local runtime logs, then retry.');
      }

      await wait(1500);
    }
  }

  async function handleAnalyze() {
    const requestedQuery = query.trim() || settings.useCaseProfile.defaultQuery;
    const profileConflict = getProfileQueryConflict(settings.useCaseProfile, requestedQuery);
    if (profileConflict) {
      setError(profileConflict);
      setErrorDetails(
        'Use-case profiles provide the analysis intent. Pick a matching profile or use the profile default query before submitting.',
      );
      return;
    }
    const profileForRequest = withEffectiveProfile(settings.useCaseProfile, requestedQuery);
    const effectiveQuery = profileForRequest.effectiveQuery ?? requestedQuery;
    if (!effectiveQuery.trim()) {
      setError('Enter a query before starting analysis.');
      return;
    }

    const backendFiles = selectedFiles.length ? [...selectedFiles] : selectedFile ? [selectedFile] : [];
    let activeMedia: MediaItem | null = selectedMedia
      ? {
        ...selectedMedia,
        useCaseProfile: profileForRequest,
        requestedQuery,
        effectiveQuery,
      }
      : null;
    if (activeMedia?.status === 'completed') {
      if (!backendFiles.length) {
        setError('Re-upload this media to run analysis again.');
        setErrorDetails('The completed result is still available, but the browser no longer has the original File object after refresh or cleanup.');
        return;
      }
      const rerunId = `${activeMedia.id}-rerun-${Date.now()}`;
      const rerunPreviewUrl = backendFiles.length === 1 && !isZipFile(backendFiles[0])
        ? URL.createObjectURL(backendFiles[0])
        : undefined;
      trackObjectUrl(rerunPreviewUrl);
      const rerunMedia: MediaItem = {
        ...activeMedia,
        id: rerunId,
        filename: `${activeMedia.filename} (rerun)`,
        status: 'draft',
        previewUrl: rerunPreviewUrl,
        jobId: undefined,
        selectedJobId: undefined,
        batchId: undefined,
        errorMessage: undefined,
      };
      if (backendFiles.length === 1) {
        localFilesRef.current[rerunId] = backendFiles[0];
      }
      localFileGroupsRef.current[rerunId] = backendFiles;
      selectedMediaIdRef.current = rerunId;
      setSelectedMedia(rerunMedia);
      setMediaLibrary((current) => [rerunMedia, ...current]);
      activeMedia = rerunMedia;
    } else if (activeMedia && ['queued', 'processing'].includes(activeMedia.status)) {
      setError(
        'This job is already queued or running. Select another draft job to submit.',
      );
      return;
    }
    if (activeMedia) {
      updateMediaItem(activeMedia.id, {
        useCaseProfile: profileForRequest,
        requestedQuery,
        effectiveQuery,
      });
      setMediaAnalysisActive(activeMedia.id, true);
    }
    setError(null);
    setAnalysisResult(null);
    setHighlightedFrameId(null);
    setJobStatus(null);
    setErrorDetails(null);
    setBatchStatus(null);
    setSelectedBatchJobId(null);

    try {
      if (backendConnected && backendFiles.length) {
        setAnalysisMode('backend');
        setActiveStep(0);
        if (activeMedia) updateMediaStatus(activeMedia.id, 'queued');
        if (isBatchSelection(backendFiles)) {
          const batch = await runBackendBatchAnalysis({
            files: backendFiles,
            query: effectiveQuery,
            fps: settings.fps,
            topK: settings.topK,
            enableVlm: shouldRequestVlm(settings),
            vlmProfile: settings.vlmProfile,
            vlmEnabled: shouldRequestVlm(settings),
            device: settings.deviceMode,
            useCaseProfile: profileForRequest,
          });
          if (!activeMedia || selectedMediaIdRef.current === activeMedia.id) setBatchStatus(batch);
          if (activeMedia) updateMediaJobReference(activeMedia.id, { batchId: batch.batchId });
          if (activeMedia) {
            rememberCachedEntry(buildCacheEntry({
              media: activeMedia,
              batchStatus: batch,
              status: batch.status,
              queryText: effectiveQuery,
            }));
          }
          const finalBatch = await pollBackendBatch(batch.batchId, activeMedia ?? undefined);
          const completedFile = finalBatch.acceptedFiles.find((file) => file.status === 'completed');
          if (!completedFile) {
            throw new Error('Batch analysis finished without a completed video result.');
          }
          const result = await getJobResult(completedFile.jobId);
          result.settings = { ...settings, useCaseProfile: profileForRequest };
          result.media.useCaseProfile = profileForRequest;
          result.media.requestedQuery = requestedQuery;
          result.media.effectiveQuery = effectiveQuery;
          if (activeMedia) {
            const jobEntry: CachedResultEntry = {
              cacheKey: jobCacheKey(completedFile.jobId),
              cacheVersion: 1,
              mediaId: activeMedia.id,
              mediaName: completedFile.filename,
              query: effectiveQuery,
              source: 'backend',
              status: 'completed',
              savedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
              jobId: completedFile.jobId,
              batchId: finalBatch.batchId,
              result,
              batchStatus: finalBatch,
            };
            rememberCachedEntry(jobEntry);
            rememberCachedEntry(buildCacheEntry({
              media: activeMedia,
              result,
              batchStatus: finalBatch,
              selectedJobId: completedFile.jobId,
              status: finalBatch.status,
              queryText: effectiveQuery,
            }));
            updateMediaJobReference(activeMedia.id, {
              batchId: finalBatch.batchId,
              selectedJobId: completedFile.jobId,
              jobId: completedFile.jobId,
            });
            updateMediaStatus(activeMedia.id, 'completed');
          }
          if (!activeMedia || selectedMediaIdRef.current === activeMedia.id) {
            setSelectedBatchJobId(completedFile.jobId);
            setActiveStep(ANALYSIS_STEPS.length);
            setAnalysisResult(result);
          }
          return;
        }

        const job = await runBackendAnalysis({
          file: backendFiles[0],
          query: effectiveQuery,
          fps: settings.fps,
          topK: settings.topK,
          enableVlm: shouldRequestVlm(settings),
          vlmProfile: settings.vlmProfile,
          vlmEnabled: shouldRequestVlm(settings),
          device: settings.deviceMode,
          useCaseProfile: profileForRequest,
        });
        if (activeMedia) updateMediaJobReference(activeMedia.id, { jobId: job.jobId, selectedJobId: job.jobId });
        const queuedStatus: JobStatus = {
          ...job,
          progress: 0,
          progressPercent: 0,
          stage: 'queued',
          message: 'Queued for analysis',
          currentStep: 'Queued for analysis',
          error: null,
        };
        if (!activeMedia || selectedMediaIdRef.current === activeMedia.id) setJobStatus(queuedStatus);
        if (activeMedia) {
          rememberCachedEntry(buildCacheEntry({
            media: activeMedia,
            jobStatus: queuedStatus,
            status: queuedStatus.status,
            queryText: effectiveQuery,
          }));
        }
        const completedStatus = await pollBackendJob(job.jobId, activeMedia ?? undefined);
        const result = await getJobResult(job.jobId);
        result.settings = { ...settings, useCaseProfile: profileForRequest };
        result.media.useCaseProfile = profileForRequest;
        result.media.requestedQuery = requestedQuery;
        result.media.effectiveQuery = effectiveQuery;
        if (activeMedia) {
          rememberCachedEntry(buildCacheEntry({
            media: activeMedia,
            result,
            jobStatus: completedStatus,
            status: 'completed',
            queryText: effectiveQuery,
          }));
          updateMediaStatus(activeMedia.id, 'completed');
        }
        if (!activeMedia || selectedMediaIdRef.current === activeMedia.id) {
          setActiveStep(ANALYSIS_STEPS.length);
          setAnalysisResult(result);
        }
        return;
      }

      if (canUsePreview && selectedMedia) {
        setAnalysisMode('preview');
        for (let index = 0; index < ANALYSIS_STEPS.length; index += 1) {
          setActiveStep(index);
          await wait(330);
        }
        const result = await runMockAnalysis({
          query: effectiveQuery,
          media: { ...selectedMedia, useCaseProfile: profileForRequest, requestedQuery, effectiveQuery },
          settings: { ...settings, useCaseProfile: profileForRequest },
        });
        result.media.useCaseProfile = profileForRequest;
        result.media.requestedQuery = requestedQuery;
        result.media.effectiveQuery = effectiveQuery;
        setActiveStep(ANALYSIS_STEPS.length);
        setAnalysisResult(result);
        rememberCachedEntry(buildCacheEntry({
          media: selectedMedia,
          result,
          source: 'preview',
          status: 'completed',
          queryText: effectiveQuery,
        }));
        return;
      }

      if (backendConnected) {
        throw new Error('Select a local image or video before running backend analysis.');
      }

      throw new Error('SafeTrace backend is not running or not reachable.');
    } catch (err) {
      if (!activeMedia || selectedMediaIdRef.current === activeMedia.id) setAnalysisMode(null);
      if (activeMedia) {
        updateMediaItem(activeMedia.id, {
          status: 'error',
          errorMessage: err instanceof Error ? err.message : 'Analysis could not be completed.',
        });
      }
      if (!activeMedia || selectedMediaIdRef.current === activeMedia.id) {
        setError(err instanceof Error ? err.message : 'Analysis could not be completed. Please try again.');
        setErrorDetails(
          err instanceof BackendJobFailureError
            ? err.debugDetails
            : err instanceof Error
              ? err.message
              : 'Analysis could not be completed. Please try again.',
        );
      }
    } finally {
      if (activeMedia) setMediaAnalysisActive(activeMedia.id, false);
    }
  }

  function handleReset() {
    setAnalysisResult(null);
    setError(null);
    setErrorDetails(null);
    setQuery(settings.useCaseProfile.defaultQuery);
    setHighlightedFrameId(null);
    setJobStatus(null);
    setBatchStatus(null);
    setSelectedBatchJobId(null);
    setAnalysisMode(null);
  }

  async function handleSelectBatchResult(jobId: string) {
    if (batchResultLoadingJobId === jobId) return;
    setBatchResultLoadingJobId(jobId);
    setError(null);
    setErrorDetails(null);
    const cached = cachedEntries[jobCacheKey(jobId)];
    if (cached?.result) {
      setSelectedBatchJobId(jobId);
      setAnalysisResult(cached.result);
      setAnalysisMode(cached.source);
      if (isCacheEntryStale(cached)) {
        setCacheMessage('Showing an older cached batch result while SafeTrace refreshes from the local runtime.');
      }
    }
    try {
      const result = await getJobResult(jobId);
      const resultProfile = result.settings?.useCaseProfile ?? result.media.useCaseProfile ?? selectedMedia?.useCaseProfile ?? settings.useCaseProfile;
      result.settings = { ...settings, useCaseProfile: resultProfile };
      result.media.useCaseProfile = resultProfile;
      result.media.requestedQuery = selectedMedia?.requestedQuery ?? result.query;
      result.media.effectiveQuery = result.query;
      setSelectedBatchJobId(jobId);
      setAnalysisResult(result);
      setActiveStep(ANALYSIS_STEPS.length);
      if (selectedMedia) {
        rememberCachedEntry({
          cacheKey: jobCacheKey(jobId),
          cacheVersion: 1,
          mediaId: selectedMedia.id,
          mediaName: selectedMedia.filename,
          query: result.query,
          source: 'backend',
          status: 'completed',
          savedAt: cached?.savedAt ?? new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          jobId,
          batchId: batchStatus?.batchId,
          result,
          batchStatus,
        });
        rememberCachedEntry(buildCacheEntry({
          media: {
            ...selectedMedia,
            useCaseProfile: resultProfile,
            requestedQuery: result.media.requestedQuery,
            effectiveQuery: result.query,
          },
          result,
          batchStatus,
          selectedJobId: jobId,
          status: batchStatus?.status ?? 'completed',
          queryText: result.query,
        }));
      }
    } catch (err) {
      if (!cached?.result) {
        setError(err instanceof Error ? err.message : 'Could not load this batch result.');
      } else {
        setCacheMessage('Showing cached batch result; the local runtime could not refresh it right now.');
      }
    } finally {
      setBatchResultLoadingJobId(null);
    }
  }

  async function deleteKnownBackendJobsForCacheEntries(entries: CachedResultEntry[]) {
    if (!backendConnected) {
      return { deleted: 0, failed: 0, skipped: collectTerminalCachedJobIds(entries).length, backendUnavailable: true };
    }
    let deleted = 0;
    let failed = 0;
    let skipped = 0;
    for (const jobId of collectTerminalCachedJobIds(entries)) {
      try {
        const status = await getJobStatus(jobId);
        if (!TERMINAL_JOB_STATUSES.has(status.status)) {
          skipped += 1;
          continue;
        }
        await deleteJob(jobId);
        deleted += 1;
      } catch {
        failed += 1;
      }
    }
    return { deleted, failed, skipped, backendUnavailable: false };
  }

  function backendDeletionMessage(result: Awaited<ReturnType<typeof deleteKnownBackendJobsForCacheEntries>>): string {
    if (result.backendUnavailable) {
      return ' Backend runtime was unavailable, so backend job folders may still exist.';
    }
    const parts = [`Deleted ${result.deleted} completed/failed backend job${result.deleted === 1 ? '' : 's'}.`];
    if (result.skipped) parts.push(`Skipped ${result.skipped} queued/running job${result.skipped === 1 ? '' : 's'}.`);
    if (result.failed) parts.push(`${result.failed} backend deletion${result.failed === 1 ? '' : 's'} could not be completed.`);
    return ` ${parts.join(' ')}`;
  }

  async function handleClearResultCache() {
    const entries = Object.values(cachedEntries);
    clearSafeTraceResultCacheStorageKeys();
    await clearCachedResults();
    setCachedEntries({});
    const deletion = await deleteKnownBackendJobsForCacheEntries(entries);
    setCacheMessage(
      `Browser result cache cleared. Original uploaded media in the browser was not stored.`
        + backendDeletionMessage(deletion)
        + ' Generated evidence/report artifacts tied to deleted backend jobs are removed by the backend only for those scoped jobs.',
    );
  }

  async function handleClearSelectedResultCache() {
    if (!selectedMedia) return;
    const entry = cachedEntries[mediaCacheKey(selectedMedia.id)];
    const childEntry = entry?.selectedJobId ? cachedEntries[jobCacheKey(entry.selectedJobId)] : undefined;
    const entriesToDelete = [entry, childEntry].filter(Boolean) as CachedResultEntry[];
    const next = { ...cachedEntries };
    clearSafeTraceResultCacheStorageKeys();
    delete next[mediaCacheKey(selectedMedia.id)];
    await deleteCachedResult(mediaCacheKey(selectedMedia.id));
    if (entry?.selectedJobId) {
      delete next[jobCacheKey(entry.selectedJobId)];
      await deleteCachedResult(jobCacheKey(entry.selectedJobId));
    }
    setCachedEntries(next);
    setAnalysisResult(null);
    setJobStatus(null);
    setBatchStatus(null);
    setSelectedBatchJobId(null);
    const deletion = await deleteKnownBackendJobsForCacheEntries(entriesToDelete);
    setCacheMessage(
      `Cleared browser cache metadata for ${selectedMedia.filename}.`
        + backendDeletionMessage(deletion)
        + ' Queued/running jobs are never deleted by this cache action.',
    );
  }

  function handleFrameSelect(frameId: string) {
    setHighlightedFrameId(frameId);
    window.requestAnimationFrame(() => {
      document.getElementById(`frame-${frameId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
    window.setTimeout(() => setHighlightedFrameId(null), 2200);
  }

  const selectedProfile = settings.useCaseProfile;
  const queryConflict = getProfileQueryConflict(selectedProfile, query);
  const effectiveQueryPreview = buildEffectiveProfileQuery(selectedProfile, query.trim() || selectedProfile.defaultQuery);
  const selectedMediaStatus = selectedMedia?.status;
  const selectedMediaBusy = selectedMediaStatus === 'queued' || selectedMediaStatus === 'processing';
  const analyzeDisabledReason = !backendConnected && !previewMode
    ? 'Start SafeTrace Local Runtime, then reconnect before analysis.'
    : !selectedFiles.length && !selectedFile && !canUsePreview
      ? 'Select a local image, video, ZIP archive, or video batch before analysis.'
    : selectedMediaBusy
      ? 'This selected job is already queued or running. Select another draft job to submit it.'
      : selectedMediaStatus === 'completed' && !selectedFiles.length && !selectedFile
        ? 'Re-upload this media to run analysis again.'
          : queryConflict
            ? queryConflict
            : !effectiveQueryPreview.trim()
              ? 'Enter a query before analysis.'
              : undefined;
  const canAnalyze = !analyzeDisabledReason;
  const analyzeButtonLabel = isLoading
    ? 'Running'
    : selectedMediaStatus === 'completed'
      ? 'Run again'
    : hasAnyActiveAnalysis
      ? 'Queue job'
      : 'Send';
  const selectedViewerJobId = (
    isJobId(analysisResult?.jobId)
      ? analysisResult?.jobId
      : isJobId(selectedBatchJobId)
        ? selectedBatchJobId
        : isJobId(jobStatus?.jobId)
          ? jobStatus?.jobId
          : undefined
  );

  return (
    <AppShell
      sidebar={
        <Sidebar
          settings={settings}
          onSettingsChange={handleSettingsChange}
          backendState={backendState}
          apiBase={activeApiBase}
          systemStatus={systemStatus}
          backendMessage={backendMessage}
          previewMode={previewMode}
        />
      }
      rightPanel={
        <VideoQueue
          mediaLibrary={mediaLibrary}
          selectedMedia={selectedMedia}
          onSelectMedia={handleSelectMedia}
          onDeleteMedia={handleDeleteMedia}
          onUploadClick={() => setIsUploadModalOpen(true)}
          uploadDisabled={controlsLocked}
          activeAnalysisMediaIds={activeAnalysisMediaIds}
        />
      }
    >
      <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <p className="text-sm font-semibold text-safety-teal">SafeTrace dashboard</p>
            <h1 className="mt-1 text-3xl font-bold tracking-normal text-slate-950">
              Safety Violation Detection
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              Upload safety footage, describe what to check for, and review evidence-backed violation findings.
            </p>
            {previewMode ? (
              <p className="mt-3 inline-flex rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-bold uppercase text-amber-700">
                Developer Preview Mode
              </p>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setActiveView('analysis')}
              className={`focus-ring inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-semibold transition ${
                activeView === 'analysis'
                  ? 'border-safety-blue bg-blue-50 text-safety-blue'
                  : 'border-slate-300 bg-white text-slate-700 hover:border-safety-blue hover:text-safety-blue'
              }`}
            >
              <ClipboardCheck className="h-4 w-4" aria-hidden="true" />
              Analysis
            </button>
            <button
              type="button"
              onClick={() => setActiveView('insights')}
              className={`focus-ring inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-semibold transition ${
                activeView === 'insights'
                  ? 'border-safety-blue bg-blue-50 text-safety-blue'
                  : 'border-slate-300 bg-white text-slate-700 hover:border-safety-blue hover:text-safety-blue'
              }`}
            >
              <BarChart3 className="h-4 w-4" aria-hidden="true" />
              Safety Insights
            </button>
          </div>
        </div>
      </header>

      {!backendConnected && !previewMode ? (
        <BackendUnavailableState
          apiBase={activeApiBase}
          apiBaseCandidates={SAFETRACE_API_BASE_CANDIDATES}
          state={backendState}
          message={backendMessage}
          onRetry={refreshBackendConnection}
        />
      ) : null}

      <ResultCachePanel
        entryCount={cachedEntryCount}
        message={cacheMessage}
        hasSelectedCachedResult={Boolean(selectedMedia && cachedEntries[mediaCacheKey(selectedMedia.id)])}
        selectedMediaName={selectedMedia?.filename}
        onClearAll={() => void handleClearResultCache()}
        onClearSelected={() => void handleClearSelectedResultCache()}
      />

      {activeView === 'insights' ? (
        <SafetyInsightsDashboard
          entries={Object.values(cachedEntries)}
          currentResult={analysisResult}
          backendConnected={backendConnected}
          onOpenResult={handleOpenDashboardEntry}
          onBackToAnalysis={() => setActiveView('analysis')}
        />
      ) : (
        <>
      <SelectedMediaViewer
        media={selectedMedia}
        disabled={controlsLocked}
        backendConnected={backendConnected}
        previewMode={previewMode}
        jobId={selectedViewerJobId}
        onUploadClick={() => setIsUploadModalOpen(true)}
      />

      <QueryTabs
        query={query}
        isLoading={isLoading}
        hasResult={Boolean(analysisResult)}
        useCaseProfile={settings.useCaseProfile}
        effectiveQuery={effectiveQueryPreview}
        queryConflict={queryConflict}
        buttonLabel={analyzeButtonLabel}
        onQueryChange={setQuery}
        onAnalyze={handleAnalyze}
        onReset={handleReset}
        canAnalyze={canAnalyze}
        disabledReason={analyzeDisabledReason}
        previewMode={previewMode && !backendConnected}
      />

      {isLoading ? (
        <AnalysisProgress
          steps={ANALYSIS_STEPS}
          activeStep={activeStep}
          currentStep={jobStatus?.currentStep}
          progress={jobStatus?.progress}
          progressPercent={jobStatus?.progressPercent}
          stage={jobStatus?.stage}
          message={jobStatus?.message}
          mode={analysisMode}
          status={jobStatus?.status}
          createdAt={jobStatus?.createdAt}
          queuedAt={jobStatus?.queuedAt}
          startedAt={jobStatus?.startedAt}
          completedAt={jobStatus?.completedAt}
          failedAt={jobStatus?.failedAt}
          cancelledAt={jobStatus?.cancelledAt}
          elapsedSeconds={jobStatus?.elapsedSeconds}
          queueWaitSeconds={jobStatus?.queueWaitSeconds}
          analysisRuntimeSeconds={jobStatus?.analysisRuntimeSeconds}
          updatedAt={jobStatus?.updatedAt}
          heartbeatAt={jobStatus?.heartbeatAt}
        />
      ) : null}
      {batchStatus ? (
        <BatchStatusPanel
          batch={batchStatus}
          selectedJobId={selectedBatchJobId}
          loadingJobId={batchResultLoadingJobId}
          onSelectJob={handleSelectBatchResult}
        />
      ) : null}
      {error ? <ErrorState message={error} details={errorDetails || jobStatus?.error || backendMessage} /> : null}

      {!analysisResult && !isLoading && !error && (backendConnected || previewMode) ? (
        <PreAnalysisState hasMedia={Boolean(selectedFiles.length || selectedFile || canUsePreview)} previewMode={canUsePreview} />
      ) : null}

      {analysisResult && !isLoading ? (
        <>
          <AnalysisSummary result={analysisResult} showExplanations={settings.visualExplanations} />
          
          <ViolationSummary 
            result={analysisResult} 
            onFrameSelect={handleFrameSelect} 
            timelineComponent={
              <TimelineVisualization 
                result={analysisResult} 
                onFrameSelect={handleFrameSelect}
                selectedFrameId={highlightedFrameId}
                hoveredFrameId={hoveredFrameId}
                onHover={setHoveredFrameId}
              />
            } 
            statisticsComponent={<StatisticsPanel result={analysisResult} />} 
          />
          
          <EvidenceFrames
            frames={analysisResult.frames}
            showExplanations={settings.visualExplanations}
            highlightedFrameId={highlightedFrameId}
            jobId={analysisResult.jobId}
            useCaseProfile={analysisResult.settings?.useCaseProfile ?? analysisResult.media.useCaseProfile}
            effectiveQuery={analysisResult.query}
            analysisDiagnostics={resultComponentDiagnostics(analysisResult)}
          />
          
          {showAnnotation && (
            <AnnotationViewer
              mediaUrl={analysisResult.frames[0]?.imageUrl}
              mediaId={analysisResult.media.id}
              mediaType={analysisResult.media.type === 'video' ? 'video' : 'image'}
            />
          )}
          
          <ReportActions result={analysisResult} />
        </>
      ) : null}
        </>
      )}

      {isUploadModalOpen && (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl bg-white p-6 shadow-xl relative">
         <button 
           onClick={() => setIsUploadModalOpen(false)}
           className="absolute right-4 top-4 text-slate-400 hover:text-slate-600"
         >
           Close
         </button>
         <h2 className="text-lg font-bold mb-4">Upload More Videos</h2>
         {/* Insert your UploadPanel component here instead of the main view */}
         <UploadPanel media={selectedMedia} onFileSelected={(file) => {
            handleFileSelected(file);
            setIsUploadModalOpen(false); // Close modal after upload
         }} onFilesSelected={(files) => {
            handleFilesSelected(files);
            setIsUploadModalOpen(false); // Close modal after upload
         }} disabled={controlsLocked} />
      </div>
    </div>
  )}

      <SafeTraceAssistant
        backendConnected={backendConnected}
        result={analysisResult}
        batch={batchStatus}
        selectedJobId={selectedBatchJobId}
      />
    </AppShell>
  );
}

function BackendUnavailableState({
  apiBase,
  apiBaseCandidates,
  state,
  message,
  onRetry,
}: {
  apiBase: string;
  apiBaseCandidates: string[];
  state: BackendConnectionState;
  message: string | null;
  onRetry: () => void;
}) {
  const isConnecting = state === 'connecting' || state === 'live';
  const title = state === 'incompatible'
    ? 'SafeTrace Local Runtime incompatible'
    : state === 'connecting'
      ? 'Connecting to SafeTrace Local Runtime'
      : 'SafeTrace Local Runtime not connected';

  return (
    <section className="rounded-lg border border-amber-200 bg-amber-50 p-6 text-amber-950 shadow-soft">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-700">
            <Server className="h-6 w-6" aria-hidden="true" />
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-amber-700">Live website loaded</p>
            <h2 className="mt-1 text-base font-bold">{title}</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6">
              To use analysis, run SafeTrace.exe on this computer, keep the local runtime window open, then return here
              and click Reconnect.
            </p>
            <p className="mt-2 max-w-3xl text-sm leading-6">
              During development, run <span className="font-mono">scripts\start_safetrace_windows.bat</span> instead.
            </p>
            <dl className="mt-3 grid gap-1 text-xs">
              <div>
                <dt className="font-semibold uppercase text-amber-700">Active API base</dt>
                <dd className="break-all font-mono text-amber-900">{apiBase}</dd>
              </div>
              <div>
                <dt className="font-semibold uppercase text-amber-700">Discovery candidates</dt>
                <dd className="break-all font-mono text-amber-900">{apiBaseCandidates.join(', ')}</dd>
              </div>
              {message ? (
                <div>
                  <dt className="font-semibold uppercase text-amber-700">Status</dt>
                  <dd>{message}</dd>
                </div>
              ) : null}
            </dl>
          </div>
        </div>
        <button
          type="button"
          onClick={onRetry}
          disabled={isConnecting}
          className="focus-ring inline-flex items-center justify-center gap-2 rounded-lg bg-amber-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-amber-800 disabled:opacity-60"
        >
          <RefreshCcw className={`h-4 w-4 ${isConnecting ? 'animate-spin' : ''}`} aria-hidden="true" />
          Reconnect to Local Runtime
        </button>
      </div>
    </section>
  );
}

function BatchStatusPanel({
  batch,
  selectedJobId,
  loadingJobId,
  onSelectJob,
}: {
  batch: BatchStatus;
  selectedJobId: string | null;
  loadingJobId: string | null;
  onSelectJob: (jobId: string) => void;
}) {
  const completed = batch.acceptedFiles.filter((file) => file.status === 'completed').length;
  const total = batch.acceptedFiles.length;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-soft">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-bold uppercase text-safety-blue">Batch analysis</p>
          <h2 className="mt-1 text-base font-bold text-slate-950">{batch.sourceFilename}</h2>
          <p className="mt-1 text-sm text-slate-600">
            {completed} of {total} accepted video{total === 1 ? '' : 's'} completed.
          </p>
        </div>
        <span className="inline-flex w-fit rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-bold uppercase text-slate-700">
          {batch.status}
        </span>
      </div>

      {batch.acceptedFiles.length ? (
        <div className="mt-4 grid gap-2">
          {batch.acceptedFiles.map((file) => {
            const isSelected = selectedJobId === file.jobId;
            const isLoading = loadingJobId === file.jobId;

            return (
              <div
                key={file.jobId}
                className={`flex flex-col gap-2 rounded-lg border px-3 py-2 sm:flex-row sm:items-center sm:justify-between ${
                  isSelected ? 'border-safety-blue bg-blue-50' : 'border-slate-100 bg-slate-50'
                }`}
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-900">{file.filename}</p>
                  <p className="text-xs text-slate-500">{formatFileSize(file.sizeBytes)}</p>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                    <span className="font-semibold uppercase">Job</span>
                    <code className="rounded bg-white px-1.5 py-0.5 font-mono text-slate-800" title={file.jobId}>
                      {formatShortJobId(file.jobId)}
                    </code>
                    <button
                      type="button"
                      onClick={() => void copyJobIdToClipboard(file.jobId)}
                      className="focus-ring inline-flex items-center gap-1 rounded border border-slate-200 bg-white px-1.5 py-0.5 font-semibold text-slate-600 transition hover:border-safety-blue hover:text-safety-blue"
                      aria-label="Copy batch job ID"
                      title={`Copy full job ID ${file.jobId}`}
                    >
                      <Copy className="h-3 w-3" aria-hidden="true" />
                      Copy job ID
                    </button>
                  </div>
                  {file.status === 'failed' ? (
                    <p className="mt-1 text-xs font-medium text-red-700">
                      {file.error || 'Analysis failed for this video.'}
                    </p>
                  ) : null}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="text-xs font-bold uppercase text-slate-600">{file.status}</span>
                  {file.status === 'completed' ? (
                    <button
                      type="button"
                      onClick={() => onSelectJob(file.jobId)}
                      disabled={isLoading}
                      className={`focus-ring rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                        isSelected
                          ? 'border-safety-blue bg-white text-safety-blue'
                          : 'border-slate-300 bg-white text-slate-700 hover:border-safety-blue hover:text-safety-blue'
                      } disabled:opacity-60`}
                    >
                      {isLoading ? 'Loading' : isSelected ? 'Viewing' : 'Open evidence'}
                    </button>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

      {batch.rejectedFiles.length ? (
        <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3">
          <p className="text-xs font-bold uppercase text-amber-800">Rejected files</p>
          <ul className="mt-2 space-y-1 text-sm text-amber-950">
            {batch.rejectedFiles.map((file) => (
              <li key={`${file.filename}-${file.reason}`}>
                <span className="font-semibold">{file.filename}</span>: {file.reason}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function ResultCachePanel({
  entryCount,
  message,
  hasSelectedCachedResult,
  selectedMediaName,
  onClearAll,
  onClearSelected,
}: {
  entryCount: number;
  message: string | null;
  hasSelectedCachedResult: boolean;
  selectedMediaName?: string;
  onClearAll: () => void;
  onClearSelected: () => void;
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-700 shadow-soft">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-safety-blue">
            <Database className="h-5 w-5" aria-hidden="true" />
          </div>
          <div>
            <p className="font-bold text-slate-950">Local result cache</p>
            <p className="mt-1 leading-6">
              Browser result cache is session-only by default and clears on refresh to avoid stale local memory.
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Session state can include job and batch IDs, result JSON, evidence metadata, backend media/report URLs, and timestamps.
              Not stored: raw uploaded videos, copied evidence image bytes, model files, credentials, or secrets.
              Clear cache removes SafeTrace browser keys immediately and, when the backend is connected, best-effort deletes only known completed/failed backend jobs from this UI session. Running or queued jobs are skipped; uploads, model assets, release archives, and unrelated data folders are never broadly deleted.
            </p>
            {message ? <p className="mt-2 text-xs font-semibold text-safety-blue">{message}</p> : null}
          </div>
        </div>
        <div className="flex flex-wrap gap-2 lg:justify-end">
          <span className="inline-flex items-center rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold uppercase text-slate-600">
            {entryCount} cached item{entryCount === 1 ? '' : 's'}
          </span>
          {hasSelectedCachedResult ? (
            <button
              type="button"
              onClick={onClearSelected}
              className="focus-ring inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-red-300 hover:text-red-700"
              title={selectedMediaName ? `Clear cached result for ${selectedMediaName}` : 'Clear selected cached result'}
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
              Clear selected browser cache
            </button>
          ) : null}
          <button
            type="button"
            onClick={onClearAll}
            disabled={entryCount === 0}
            className="focus-ring inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            Clear browser result cache
          </button>
        </div>
      </div>
    </section>
  );
}

function ErrorState({ message, details }: { message: string; details?: string | null }) {
  return (
    <section className="rounded-lg border border-red-200 bg-red-50 p-5 text-red-900">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5" aria-hidden="true" />
        <div>
          <h2 className="text-sm font-bold">{message}</h2>
          <details className="mt-2 text-sm">
            <summary className="cursor-pointer font-semibold">Debug details</summary>
            <p className="mt-2 text-red-800">{details || 'The local analysis did not complete.'}</p>
          </details>
        </div>
      </div>
    </section>
  );
}

function PreAnalysisState({ hasMedia, previewMode }: { hasMedia: boolean; previewMode: boolean }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 text-slate-700 shadow-soft">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-safety-teal text-white">
          {hasMedia ? <ClipboardCheck className="h-6 w-6" aria-hidden="true" /> : <UploadCloud className="h-6 w-6" aria-hidden="true" />}
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-950">
            {hasMedia ? 'Ready to run analysis' : 'Select local media to begin'}
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-6">
            {previewMode
              ? 'Developer Preview Mode is enabled. Sample media can generate preview findings without the backend.'
              : hasMedia
                ? 'The selected media and query are ready. Click Analyze to submit the file to the local SafeTrace backend.'
                : 'Upload a local image, video, ZIP archive, or batch of videos, then describe what SafeTrace should inspect.'}
          </p>
        </div>
      </div>
    </section>
  );
}

export default App;
