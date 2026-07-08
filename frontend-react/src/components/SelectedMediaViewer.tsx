import { Archive, Clock, Copy, FileImage, FileVideo, HardDrive, UploadCloud } from 'lucide-react';
import type { MediaItem } from '../types/analysis';
import { supportLevelLabel } from '../data/useCaseProfiles';
import { copyJobIdToClipboard, formatShortJobId } from '../utils/jobIds';
import { StatusBadge } from './StatusBadge'; // Make sure this is imported

type SelectedMediaViewerProps = {
  media: MediaItem | null;
  disabled?: boolean;
  backendConnected?: boolean;
  previewMode?: boolean;
  jobId?: string | null;
  onUploadClick?: () => void;
};

export function SelectedMediaViewer({
  media,
  disabled = false,
  backendConnected = false,
  previewMode = false,
  jobId,
  onUploadClick,
}: SelectedMediaViewerProps) {
  if (!media) {
    return (
      <div className="mb-6 overflow-hidden rounded-xl border border-slate-200 bg-white p-6 shadow-soft">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-slate-500">
              <UploadCloud className="h-6 w-6" aria-hidden="true" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-950">No local media selected</h2>
              <p className="mt-1 text-sm leading-6 text-slate-600">
                {disabled
                  ? 'Start the local backend before choosing footage.'
                  : 'Upload an image or video to send to the SafeTrace backend.'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onUploadClick}
            disabled={disabled}
            className="focus-ring inline-flex items-center justify-center gap-2 rounded-lg bg-safety-blue px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:opacity-50"
          >
            <UploadCloud className="h-4 w-4" aria-hidden="true" />
            Upload media
          </button>
        </div>
      </div>
    );
  }

  const Icon = media.type === 'video' ? FileVideo : media.type === 'image' ? FileImage : Archive;
  const resultJobId = jobId || media.selectedJobId || media.jobId;
  const useCaseProfile = media.useCaseProfile;
  const statusDot = media.status === 'error'
    ? 'bg-red-500'
    : media.status === 'queued'
      ? 'bg-amber-500'
      : media.status === 'processing'
        ? 'bg-blue-500'
        : media.status === 'draft'
          ? 'bg-slate-400'
          : 'bg-emerald-500';

  return (
    <div className="mb-6 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-soft">
      <div className="flex flex-col lg:flex-row relative">
        
        <div className="flex aspect-video w-full shrink-0 items-center justify-center bg-slate-900 lg:w-1/3 xl:w-1/4">
          {media.previewUrl ? (
             media.type === 'video' ? (
               <video src={media.previewUrl} className="h-full w-full object-cover" controls muted />
             ) : media.type === 'image' ? (
               <img src={media.previewUrl} alt={media.filename} className="h-full w-full object-cover" />
             ) : (
               <Icon className="h-12 w-12 text-slate-600" />
             )
          ) : (
             <div className="flex max-w-[80%] flex-col items-center gap-2 text-center">
               <Icon className="h-12 w-12 text-slate-600" />
               {media.status === 'completed' ? (
                 <p className="text-xs font-medium leading-5 text-slate-400">
                   Preview unavailable after refresh; re-upload this media to preview or run analysis again.
                 </p>
               ) : null}
             </div>
          )}
        </div>

        <div className="flex flex-1 flex-col justify-center border-b border-slate-100 p-5 lg:border-b-0 lg:border-r border-slate-100 relative">
          
          <div className="absolute right-5 top-5">
             <StatusBadge
               label={previewMode && media.source === 'sample'
                 ? 'Preview sample'
                 : backendConnected
                   ? 'Backend ready'
                   : 'Local file selected'}
               tone={backendConnected || previewMode ? 'success' : 'warning'}
             />
          </div>

          <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-safety-blue">
            Selected for Analysis
          </div>
          <h2 className="text-lg font-bold text-slate-950 truncate pr-32">{media.filename}</h2>
          
          <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-slate-600">
            <div className="flex items-center gap-1.5">
              <HardDrive className="h-4 w-4" /> {media.sizeLabel}
            </div>
            {media.duration && (
              <div className="flex items-center gap-1.5">
                <Clock className="h-4 w-4" /> {media.duration}
              </div>
            )}
            <div className="flex items-center gap-1.5 capitalize">
              <span className={`flex h-2 w-2 rounded-full ${statusDot}`}></span>
              Status: {media.status}
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2 border-t border-slate-100 pt-4 text-xs text-slate-500">
            <HardDrive className="h-4 w-4" aria-hidden="true" />
            {previewMode && media.source === 'sample'
                ? 'Developer preview media is not a backend analysis result.'
              : media.type === 'unknown'
                ? 'Bulk media is ready for local backend batch analysis.'
              : media.status === 'completed'
                ? 'Completed result remains available. Use Run again while the original browser File is still available, or re-upload after refresh.'
              : 'Selected media is ready for local backend analysis.'}
          </div>

          {useCaseProfile ? (
            <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              <span className="font-semibold uppercase text-slate-500">Use-case profile</span>
              <span className="ml-2 font-semibold text-slate-900">{useCaseProfile.label}</span>
              <span className="ml-2 rounded-full border border-slate-200 bg-white px-2 py-0.5 font-semibold uppercase text-slate-500">
                {supportLevelLabel(useCaseProfile.backendSupportLevel)}
              </span>
              <p className="mt-1 leading-5">{useCaseProfile.description}</p>
              {media.effectiveQuery || useCaseProfile.effectiveQuery ? (
                <p className="mt-1 leading-5">
                  Effective query: <span className="font-semibold text-slate-900">{media.effectiveQuery ?? useCaseProfile.effectiveQuery}</span>
                </p>
              ) : null}
              {useCaseProfile.limitations ? (
                <p className="mt-1 leading-5 text-amber-800">{useCaseProfile.limitations}</p>
              ) : null}
            </div>
          ) : null}

          {resultJobId ? (
            <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              <span className="font-semibold uppercase text-slate-500">Result job</span>
              <code className="rounded bg-white px-2 py-1 font-mono text-[11px] font-semibold text-slate-900" title={resultJobId}>
                {formatShortJobId(resultJobId)}
              </code>
              <button
                type="button"
                onClick={() => void copyJobIdToClipboard(resultJobId)}
                className="focus-ring inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 font-semibold text-slate-700 transition hover:border-safety-blue hover:text-safety-blue"
                aria-label="Copy selected media job ID"
                title={`Copy full job ID ${resultJobId}`}
              >
                <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                Copy job ID
              </button>
            </div>
          ) : null}
        </div>

      </div>
    </div>
  );
}
