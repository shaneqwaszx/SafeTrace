import { Archive, Clock, Copy, Eye, Edit3, Trash2, HardDrive, Upload, CheckCircle2, LoaderCircle, AlertCircle, FileVideo, FileImage } from 'lucide-react';
import type { MediaItem, MediaStatus } from '../types/analysis';
import { formatDateTime } from '../utils/formatters';
import { copyJobIdToClipboard, formatShortJobId } from '../utils/jobIds';

type VideoQueueProps = {
  mediaLibrary: MediaItem[];
  selectedMedia: MediaItem | null;
  onSelectMedia: (media: MediaItem) => void;
  onDeleteMedia?: (mediaId: string) => void;
  onPreviewMedia?: (media: MediaItem) => void;
  onUploadClick?: () => void;
  uploadDisabled?: boolean;
  activeAnalysisMediaIds?: Record<string, true>;
};

function StatusIcon({ status }: { status: MediaStatus }) {
  switch (status) {
    case 'draft':
      return <Edit3 className="h-4 w-4 text-slate-500" />;
    case 'completed':
      return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
    case 'processing':
      return <LoaderCircle className="h-4 w-4 animate-spin text-blue-500" />;
    case 'queued':
      return <Clock className="h-4 w-4 text-amber-500" />;
    case 'error':
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    default:
      return <Clock className="h-4 w-4 text-slate-400" />;
  }
}

function StatusBadgeVideo({ status }: { status: MediaStatus }) {
  const styles: Record<MediaStatus, string> = {
    draft: 'bg-slate-100 text-slate-700 border-slate-200',
    ready: 'bg-slate-100 text-slate-700 border-slate-200',
    queued: 'bg-amber-50 text-amber-700 border-amber-200',
    processing: 'bg-blue-50 text-blue-700 border-blue-200',
    completed: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    error: 'bg-red-50 text-red-700 border-red-200',
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase ${styles[status]}`}>
      <StatusIcon status={status} />
      {status}
    </span>
  );
}

export function VideoQueue({
  mediaLibrary,
  selectedMedia,
  onSelectMedia,
  onDeleteMedia,
  onPreviewMedia,
  onUploadClick,
  uploadDisabled = false,
  activeAnalysisMediaIds = {},
}: VideoQueueProps) {
  return (
    <div className="flex h-full flex-col gap-4 overflow-auto p-5">
      <div className="flex items-center gap-2">
        <HardDrive className="h-5 w-5 text-safety-blue" />
        <div>
          <p className="text-sm font-bold text-slate-950">Job Queue</p>
          <p className="text-xs text-slate-500">{mediaLibrary.length} job item{mediaLibrary.length !== 1 ? 's' : ''}</p>
        </div>

        <button
          type="button"
          onClick={onUploadClick}
          disabled={uploadDisabled}
          title="Ingest new video"
          className="ml-auto flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100 text-indigo-700 transition hover:bg-indigo-200 hover:text-indigo-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
        </button>
        
      </div>

      <div className="space-y-3">
        {mediaLibrary.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-500">
            {uploadDisabled ? 'Connect to the backend to add media.' : 'Upload local media to start a backend analysis.'}
          </div>
        ) : null}

        {mediaLibrary.map((media) => {
          const isSelected = media.id === selectedMedia?.id;
          const isActiveAnalysis = Boolean(activeAnalysisMediaIds[media.id]);
          const Icon = media.type === 'video' ? FileVideo : media.type === 'image' ? FileImage : Archive;
          const resultJobId = media.selectedJobId || media.jobId;

          return (
            <div
              key={media.id}
              className={`rounded-lg border-2 transition ${
                isSelected ? 'border-safety-blue bg-blue-50' : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <button
                type="button"
                onClick={() => onSelectMedia(media)}
                className="w-full p-3 text-left"
              >
                <div className="flex items-start gap-3">
                  <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${
                    isSelected ? 'bg-white text-safety-blue shadow-insetLine' : 'bg-slate-100 text-slate-600'
                  }`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-semibold text-slate-950">{media.filename}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <StatusBadgeVideo status={media.status} />
                      {isActiveAnalysis ? (
                        <span className="rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-semibold uppercase text-blue-700">
                          Active job
                        </span>
                      ) : media.status === 'queued' ? (
                        <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase text-amber-700">
                          Waiting for worker
                        </span>
                      ) : media.status === 'draft' || media.status === 'ready' ? (
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-semibold uppercase text-slate-600">
                          Draft
                        </span>
                      ) : null}
                      <span className="text-[10px] font-medium text-slate-500">{media.sizeLabel}</span>
                      {media.duration && (
                        <span className="text-[10px] font-medium text-slate-500">{media.duration}</span>
                      )}
                    </div>
                    {media.uploadedAt && (
                      <p className="mt-1 text-[10px] text-slate-400">{formatDateTime(media.uploadedAt)}</p>
                    )}
                    {media.useCaseProfile ? (
                      <p className="mt-1 text-[10px] font-semibold text-slate-500">
                        Profile: {media.useCaseProfile.label}
                      </p>
                    ) : null}
                    {media.effectiveQuery ? (
                      <p className="mt-1 line-clamp-2 text-[10px] text-slate-500">
                        Query: {media.effectiveQuery}
                      </p>
                    ) : null}
                    {media.errorMessage ? (
                      <p className="mt-1 line-clamp-2 text-[10px] font-semibold text-red-600">
                        Error: {media.errorMessage}
                      </p>
                    ) : null}
                    {(media.jobRuntimeLabel || media.requestedModeLabel || media.actualReviewLabel || media.actualDeviceLabel) ? (
                      <div className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] leading-4 text-slate-500">
                        {media.jobRuntimeLabel ? (
                          <span><span className="font-semibold text-slate-700">Runtime:</span> {media.jobRuntimeLabel}</span>
                        ) : null}
                        {media.requestedModeLabel ? (
                          <span><span className="font-semibold text-slate-700">Requested:</span> {media.requestedModeLabel}</span>
                        ) : null}
                        {media.actualReviewLabel ? (
                          <span className="col-span-2"><span className="font-semibold text-slate-700">Actual review:</span> {media.actualReviewLabel}</span>
                        ) : null}
                        {media.actualDeviceLabel ? (
                          <span><span className="font-semibold text-slate-700">Device:</span> {media.actualDeviceLabel}</span>
                        ) : null}
                      </div>
                    ) : null}
                    {(media.vlmStatusLabel || media.mobileSamStatusLabel) ? (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {media.vlmStatusLabel ? (
                          <span className="rounded-full border border-indigo-100 bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold text-indigo-700">
                            VLM: {media.vlmStatusLabel}
                          </span>
                        ) : null}
                        {media.mobileSamStatusLabel ? (
                          <span className="rounded-full border border-cyan-100 bg-cyan-50 px-2 py-0.5 text-[10px] font-semibold text-cyan-700">
                            MobileSAM: {media.mobileSamStatusLabel}
                          </span>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </div>
              </button>

              {resultJobId ? (
                <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 px-3 py-2 text-[10px] text-slate-500">
                  <span className="font-bold uppercase">Job</span>
                  <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-slate-800" title={resultJobId}>
                    {formatShortJobId(resultJobId)}
                  </code>
                  <button
                    type="button"
                    onClick={() => void copyJobIdToClipboard(resultJobId)}
                    className="focus-ring inline-flex items-center gap-1 rounded border border-slate-200 bg-white px-1.5 py-0.5 font-semibold text-slate-600 transition hover:border-safety-blue hover:text-safety-blue"
                    aria-label="Copy queue job ID"
                    title={`Copy full job ID ${resultJobId}`}
                  >
                    <Copy className="h-3 w-3" aria-hidden="true" />
                    Copy job ID
                  </button>
                </div>
              ) : media.batchId ? (
                <p className="border-t border-slate-100 px-3 py-2 text-[10px] font-semibold uppercase text-slate-500">
                  Batch {media.batchId}
                </p>
              ) : null}

              <div className="flex items-center gap-1 border-t border-slate-100 px-3 py-2">
                {onPreviewMedia && media.previewUrl && (
                  <button
                    type="button"
                    onClick={() => onPreviewMedia(media)}
                    className="focus-ring inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-semibold text-slate-600 transition hover:bg-slate-100"
                  >
                    <Eye className="h-3.5 w-3.5" /> Preview
                  </button>
                )}
                <button
                  type="button"
                  className="focus-ring inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-semibold text-slate-600 transition hover:bg-slate-100"
                >
                  <Edit3 className="h-3.5 w-3.5" /> Edit
                </button>
                {onDeleteMedia && (
                  <button
                    type="button"
                    onClick={() => onDeleteMedia(media.id)}
                    className="focus-ring ml-auto inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-semibold text-red-600 transition hover:bg-red-50"
                  >
                    <Trash2 className="h-3.5 w-3.5" /> Delete
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
