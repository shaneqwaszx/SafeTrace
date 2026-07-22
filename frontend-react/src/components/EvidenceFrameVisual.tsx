import clsx from 'clsx';
import { AlertTriangle, ImageIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Detection, FrameResult } from '../types/analysis';
import { formatConfidence } from '../utils/formatters';

type EvidenceFrameVisualProps = {
  frame: FrameResult;
};

function getBoxClasses(label: string) {
  const normalized = label.toLowerCase();

  if (normalized.includes('restricted') || normalized.includes('zone')) {
    return 'border-blue-300 bg-blue-400/10 text-blue-50';
  }

  if (normalized.includes('helmet') || normalized.includes('seatbelt') || normalized.includes('vest')) {
    return 'border-emerald-300 bg-emerald-400/10 text-emerald-50';
  }

  if (normalized.includes('head') || normalized.includes('torso') || normalized.includes('technician')) {
    return 'border-amber-300 bg-amber-400/10 text-amber-50';
  }

  return 'border-red-300 bg-red-400/10 text-red-50';
}

function getVariantDetails(variant: FrameResult['visualVariant']) {
  if (variant === 'loading-bay') {
    return {
      label: 'Loading bay camera',
      accent: 'bg-blue-400',
      shapes: (
        <>
          <div className="absolute left-[10%] top-[72%] h-[3px] w-[78%] bg-yellow-300/60" />
          <div className="absolute left-[20%] top-[20%] h-[52%] w-[58%] rounded border border-dashed border-yellow-300/50" />
          <div className="absolute right-[10%] top-[24%] h-[42%] w-[13%] rounded bg-slate-400/20" />
        </>
      ),
    };
  }

  if (variant === 'maintenance') {
    return {
      label: 'Maintenance camera',
      accent: 'bg-emerald-400',
      shapes: (
        <>
          <div className="absolute left-[8%] top-[58%] h-[2px] w-[84%] bg-white/15" />
          <div className="absolute left-[16%] top-[30%] h-[30%] w-[24%] rounded bg-slate-400/20" />
          <div className="absolute right-[14%] top-[18%] h-[20%] w-[16%] rounded border border-emerald-300/40" />
        </>
      ),
    };
  }

  return {
    label: 'Worksite camera',
    accent: 'bg-red-400',
    shapes: (
      <>
        <div className="absolute left-[12%] top-[18%] h-[56%] w-[12%] rounded bg-slate-400/20" />
        <div className="absolute left-[70%] top-[12%] h-[66%] w-[10%] rounded bg-slate-400/20" />
        <div className="absolute bottom-[12%] left-[8%] h-[2px] w-[82%] bg-white/15" />
        <div className="absolute bottom-[24%] left-[10%] h-[2px] w-[74%] bg-white/10" />
      </>
    ),
  };
}

function DetectionOverlay({ detection }: { detection: Detection }) {
  const [left, top, width, height] = detection.bbox;

  return (
    <div
      className={clsx('absolute rounded border-2', getBoxClasses(detection.label))}
      style={{
        left: `${left}%`,
        top: `${top}%`,
        width: `${width}%`,
        height: `${height}%`,
      }}
    >
      <span className="absolute left-1 top-1 rounded bg-slate-950/80 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-normal text-white">
        {detection.label}
      </span>
    </div>
  );
}

function SampleEvidenceVisual({ frame }: EvidenceFrameVisualProps) {
  const variant = getVariantDetails(frame.visualVariant);
  const topDetectionConfidence = frame.detections.length > 0
    ? Math.max(...frame.detections.map((detection) => detection.confidence))
    : 0;

  return (
    <div className="frame-surface relative aspect-video min-h-0 overflow-hidden rounded-lg">
      <div className="absolute inset-0 opacity-80">{variant.shapes}</div>

      <div className="absolute left-3 top-3 flex items-center gap-2 rounded-lg bg-slate-950/75 px-3 py-2 text-xs font-semibold text-white">
        <span className={`h-2 w-2 rounded-full ${variant.accent}`} />
        {variant.label}
      </div>

      {frame.detections.map((detection) => (
        <DetectionOverlay key={detection.id} detection={detection} />
      ))}

      <div className="absolute bottom-3 left-3 inline-flex max-w-[calc(100%-1.5rem)] flex-wrap items-center gap-2 rounded-lg bg-slate-950/80 px-3 py-2 text-xs font-semibold text-white">
        <ImageIcon className="h-4 w-4" aria-hidden="true" />
        Local analysis preview
        <span className="rounded bg-white/10 px-1.5 py-0.5">
          Top detection {formatConfidence(topDetectionConfidence)}
        </span>
      </div>
    </div>
  );
}

export function EvidenceFrameVisual({ frame }: EvidenceFrameVisualProps) {
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = Boolean(frame.imageUrl && !imageFailed);
  const shouldBlockFallback = Boolean(frame.imageUrl && frame.evidenceImageRequired && imageFailed);
  const hasBackendMissingImage = Boolean(!frame.imageUrl && frame.imageMessage);

  useEffect(() => {
    setImageFailed(false);
  }, [frame.imageUrl]);

  if (shouldBlockFallback || hasBackendMissingImage) {
    return (
      <div className="flex aspect-video min-h-0 flex-col justify-center rounded-lg border border-amber-300 bg-amber-50 p-5 text-amber-950">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
          <div>
            <p className="text-sm font-bold">Evidence image unavailable</p>
            <p className="mt-2 break-all text-sm leading-6">{frame.imageMessage || `Expected file: ${frame.imageUrl}`}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!showImage) {
    return <SampleEvidenceVisual frame={frame} />;
  }

  return (
    <figure className="overflow-hidden rounded-lg bg-slate-950">
      <div className="flex aspect-video min-h-0 items-center justify-center">
        <img
          className="h-full w-full object-contain"
          src={frame.imageUrl}
          loading="lazy"
          decoding="async"
          alt={`Annotated evidence frame ${frame.frameNumber} at ${frame.timestamp}`}
          onError={() => setImageFailed(true)}
        />
      </div>
      <figcaption className="flex flex-wrap items-center gap-2 border-t border-white/10 bg-slate-950 px-3 py-2 text-xs font-semibold text-slate-100">
        <ImageIcon className="h-4 w-4" aria-hidden="true" />
        Annotated evidence frame
        <span className="rounded bg-white/10 px-1.5 py-0.5">Frame {frame.frameNumber}</span>
      </figcaption>
    </figure>
  );
}
