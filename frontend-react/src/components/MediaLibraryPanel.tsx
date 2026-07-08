import { CalendarClock, Database, FileImage, FileVideo, HardDrive } from 'lucide-react';
import type { MediaItem } from '../types/analysis';
import { formatDateTime } from '../utils/formatters';
import { StatusBadge } from './StatusBadge';

type MediaLibraryPanelProps = {
  selectedMedia: MediaItem;
  mediaLibrary: MediaItem[];
  onSelectMedia: (media: MediaItem) => void;
};

function getStatusTone(status: MediaItem['status']) {
  if (status === 'ready' || status === 'completed') {
    return 'success';
  }

  if (status === 'processing' || status === 'queued') {
    return 'info';
  }

  if (status === 'draft') return 'neutral';

  return 'danger';
}

export function MediaLibraryPanel({ selectedMedia, mediaLibrary, onSelectMedia }: MediaLibraryPanelProps) {
  const SelectedIcon = selectedMedia.type === 'video' ? FileVideo : FileImage;

  return (
    <div className="sticky top-0 flex max-h-screen flex-col gap-5 overflow-auto p-5">
      <div>
        <p className="text-sm font-bold text-slate-950">Media panel</p>
        <p className="mt-1 text-sm leading-6 text-slate-500">Switch sample media or review the current selection.</p>
      </div>

      <section className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-950">
          <HardDrive className="h-4 w-4 text-safety-blue" aria-hidden="true" />
          Selected media
        </div>
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-safety-blue shadow-insetLine">
            <SelectedIcon className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <p className="break-words text-sm font-semibold leading-5 text-slate-950">{selectedMedia.filename}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <StatusBadge label={selectedMedia.status === 'ready' ? 'Ready' : selectedMedia.status} tone={getStatusTone(selectedMedia.status)} />
              <StatusBadge label={selectedMedia.sizeLabel} tone="neutral" />
              {selectedMedia.duration ? <StatusBadge label={selectedMedia.duration} tone="neutral" /> : null}
            </div>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-950">
          <Database className="h-4 w-4 text-safety-teal" aria-hidden="true" />
          Sample library
        </div>
        <div className="space-y-3">
          {mediaLibrary.map((media) => {
            const Icon = media.type === 'video' ? FileVideo : FileImage;
            const isSelected = media.id === selectedMedia.id;

            return (
              <button
                key={media.id}
                className={`focus-ring w-full rounded-lg border p-3 text-left transition ${
                  isSelected
                    ? 'border-safety-blue bg-blue-50 shadow-insetLine'
                    : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                }`}
                type="button"
                onClick={() => onSelectMedia(media)}
                aria-pressed={isSelected}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg shadow-insetLine ${
                      isSelected ? 'bg-white text-safety-blue' : 'bg-slate-50 text-slate-600'
                    }`}
                  >
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="break-words text-sm font-semibold leading-5 text-slate-950">{media.filename}</p>
                    <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-500">
                      <CalendarClock className="h-3.5 w-3.5" aria-hidden="true" />
                      {formatDateTime(media.uploadedAt)}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <StatusBadge label={media.status} tone={getStatusTone(media.status)} />
                      <StatusBadge label={media.sizeLabel} tone="neutral" />
                      {isSelected ? <StatusBadge label="Selected" tone="info" /> : null}
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <p className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-500">
        Preview mode: results are generated from local sample data until backend integration is connected.
      </p>
    </div>
  );
}
