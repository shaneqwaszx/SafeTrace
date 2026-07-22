import type { AnalysisResult, BatchStatus, JobStatus } from '../types/analysis';

export const RESULT_CACHE_VERSION = 2;
const DB_NAME = 'safetrace-result-cache';
const STORE_NAME = 'results';
const LOCAL_STORAGE_VERSION_KEY = 'safetrace.resultCache.version';
const RESULT_CACHE_STORAGE_PREFIXES = ['safetrace.resultCache', 'safetrace:resultCache'];
const MAX_CACHE_ENTRY_BYTES = 2_500_000;
const STALE_AFTER_MS = 7 * 24 * 60 * 60 * 1000;

export type CacheSource = 'backend' | 'preview';

export type CachedResultEntry = {
  cacheKey: string;
  cacheVersion: number;
  mediaId: string;
  mediaName: string;
  query: string;
  source: CacheSource;
  status: string;
  savedAt: string;
  updatedAt: string;
  jobId?: string;
  batchId?: string;
  selectedJobId?: string;
  executionIdentityKey?: string;
  result?: AnalysisResult;
  jobStatus?: JobStatus | null;
  batchStatus?: BatchStatus | null;
};

export type CacheSaveResult = {
  saved: boolean;
  reason?: string;
};

export function mediaCacheKey(mediaId: string): string {
  return `media:${mediaId}`;
}

export function jobCacheKey(jobId: string): string {
  return `job:${jobId}`;
}

function executionIdentityKey(result?: AnalysisResult): string | undefined {
  if (!result?.executionIdentity) return undefined;
  return JSON.stringify(result.executionIdentity, Object.keys(result.executionIdentity).sort());
}

export function cacheEntryHasValidOwnership(entry: CachedResultEntry): boolean {
  if (!entry.result) return true;
  const expected = entry.selectedJobId || entry.jobId || entry.result.jobId;
  if (!expected || entry.result.jobId !== expected) return false;
  if (entry.result.frames.some((frame) => frame.jobId !== expected)) return false;
  return entry.executionIdentityKey === executionIdentityKey(entry.result);
}

export function isCacheEntryStale(entry: CachedResultEntry): boolean {
  const updatedAt = Date.parse(entry.updatedAt || entry.savedAt);
  return Number.isFinite(updatedAt) && Date.now() - updatedAt > STALE_AFTER_MS;
}

function isBrowserOnlyUrl(value?: string | null): boolean {
  return Boolean(value && /^(blob:|data:)/i.test(value));
}

function pruneTechnicalEvidence(value: Record<string, unknown>): Record<string, unknown> {
  const { raw, ...rest } = value;
  return rest;
}

export function sanitizeResultForCache(result: AnalysisResult): AnalysisResult {
  return {
    ...result,
    media: {
      ...result.media,
      previewUrl: undefined,
    },
    frames: result.frames.map((frame) => ({
      ...frame,
      imageUrl: isBrowserOnlyUrl(frame.imageUrl) ? undefined : frame.imageUrl,
      technicalEvidence: pruneTechnicalEvidence(frame.technicalEvidence),
    })),
    events: result.events?.map((event) => ({
      ...event,
      supportingFrames: event.supportingFrames.map((frame) => ({
        ...frame,
        imageUrl: isBrowserOnlyUrl(frame.imageUrl) ? undefined : frame.imageUrl,
      })),
    })),
  };
}

function jsonByteSize(value: unknown): number {
  return new Blob([JSON.stringify(value)]).size;
}

function compactEntry(entry: CachedResultEntry): CachedResultEntry {
  if (!entry.result) return entry;
  return {
    ...entry,
    result: {
      ...entry.result,
      frames: entry.result.frames.map((frame) => ({
        ...frame,
        detections: frame.detections.slice(0, 25),
        technicalEvidence: pruneTechnicalEvidence(frame.technicalEvidence),
      })),
    },
  };
}

function prepareEntry(entry: CachedResultEntry): CachedResultEntry | null {
  const now = new Date().toISOString();
  const prepared: CachedResultEntry = {
    ...entry,
    cacheVersion: RESULT_CACHE_VERSION,
    savedAt: entry.savedAt || now,
    updatedAt: now,
    result: entry.result ? sanitizeResultForCache(entry.result) : undefined,
    executionIdentityKey: executionIdentityKey(entry.result),
  };
  if (!cacheEntryHasValidOwnership(prepared)) return null;
  if (jsonByteSize(prepared) <= MAX_CACHE_ENTRY_BYTES) return prepared;
  const compact = compactEntry(prepared);
  if (jsonByteSize(compact) <= MAX_CACHE_ENTRY_BYTES) return compact;
  return null;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, RESULT_CACHE_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'cacheKey' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  callback: (store: IDBObjectStore) => IDBRequest<T> | void,
): Promise<T | undefined> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, mode);
    const store = transaction.objectStore(STORE_NAME);
    const request = callback(store);
    let value: T | undefined;
    if (request) {
      request.onsuccess = () => {
        value = request.result;
      };
      request.onerror = () => reject(request.error);
    }
    transaction.oncomplete = () => {
      db.close();
      resolve(value);
    };
    transaction.onerror = () => {
      db.close();
      reject(transaction.error);
    };
  });
}

export async function loadCachedResults(): Promise<CachedResultEntry[]> {
  if (!('indexedDB' in window)) return [];
  const entries = await withStore<CachedResultEntry[]>('readonly', (store) => store.getAll());
  return (entries || []).filter((entry) => (
    entry.cacheVersion === RESULT_CACHE_VERSION && cacheEntryHasValidOwnership(entry)
  ));
}

export async function saveCachedResult(entry: CachedResultEntry): Promise<CacheSaveResult> {
  if (!('indexedDB' in window)) {
    return { saved: false, reason: 'IndexedDB is not available in this browser.' };
  }
  const prepared = prepareEntry(entry);
  if (!prepared) {
    return { saved: false, reason: 'Result was too large or failed browser-cache ownership validation.' };
  }
  await withStore('readwrite', (store) => store.put(prepared));
  localStorage.setItem(LOCAL_STORAGE_VERSION_KEY, String(RESULT_CACHE_VERSION));
  return { saved: true };
}

export async function deleteCachedResult(cacheKey: string): Promise<void> {
  if (!('indexedDB' in window)) return;
  await withStore('readwrite', (store) => store.delete(cacheKey));
}

export async function clearCachedResults(): Promise<void> {
  if (!('indexedDB' in window)) return;
  await withStore('readwrite', (store) => store.clear());
  clearSafeTraceResultCacheStorageKeys();
}

export function clearSafeTraceResultCacheStorageKeys(): void {
  [window.localStorage, window.sessionStorage].forEach((storage) => {
    try {
      for (let index = storage.length - 1; index >= 0; index -= 1) {
        const key = storage.key(index);
        if (key && RESULT_CACHE_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
          storage.removeItem(key);
        }
      }
    } catch {
      // Cache cleanup remains best-effort when browser storage is restricted.
    }
  });
}
