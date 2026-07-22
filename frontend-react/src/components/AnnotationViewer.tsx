import { useCallback, useEffect, useRef, useState } from 'react';
import { Pencil, Square, Type, Download, Trash2, Save, CheckCircle2, LoaderCircle } from 'lucide-react';
import type { Annotation } from '../types/analysis';

type AnnotationViewerProps = {
  mediaUrl?: string;
  mediaId: string;
  mediaType?: 'video' | 'image';
};

const ANNOTATIONS_KEY = 'safetrace_annotations';

function loadAnnotations(mediaId: string): Annotation[] {
  try {
    const raw = localStorage.getItem(ANNOTATIONS_KEY);
    if (!raw) return [];
    const all: Record<string, Annotation[]> = JSON.parse(raw);
    return all[mediaId] || [];
  } catch {
    return [];
  }
}

function saveAnnotations(mediaId: string, annotations: Annotation[]) {
  try {
    const raw = localStorage.getItem(ANNOTATIONS_KEY);
    const all: Record<string, Annotation[]> = raw ? JSON.parse(raw) : {};
    all[mediaId] = annotations;
    localStorage.setItem(ANNOTATIONS_KEY, JSON.stringify(all));
  } catch {
    console.warn('Failed to save annotations to localStorage');
  }
}

const ANNOTATION_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];

export function AnnotationViewer({ mediaUrl, mediaId, mediaType = 'image' }: AnnotationViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>(() => loadAnnotations(mediaId));
  const [mode, setMode] = useState<'none' | 'bbox' | 'note'>('none');
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(null);
  const [currentRect, setCurrentRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null);
  const [noteText, setNoteText] = useState('');

  useEffect(() => {
    const loaded = loadAnnotations(mediaId);
    setAnnotations(loaded);
  }, [mediaId]);

  useEffect(() => {
    drawCanvas();
  }, [annotations, currentRect, selectedAnnotation]);

  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    annotations.forEach((ann) => {
      const isSelected = ann.id === selectedAnnotation;
      ctx.strokeStyle = ann.color || ANNOTATION_COLORS[0];
      ctx.lineWidth = isSelected ? 3 : 2;

      if (ann.type === 'bbox' && ann.bbox) {
        const [x, y, w, h] = ann.bbox;
        ctx.strokeRect(x, y, w, h);
        if (isSelected) {
          ctx.fillStyle = (ann.color || ANNOTATION_COLORS[0]) + '20';
          ctx.fillRect(x, y, w, h);
        }
        ctx.fillStyle = ann.color || ANNOTATION_COLORS[0];
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(ann.label || 'Annotation', x + 4, y - 4);
      }

      if (ann.type === 'note' && ann.note && ann.bbox) {
        const [x, y] = ann.bbox;
        ctx.fillStyle = ann.color || ANNOTATION_COLORS[0];
        ctx.font = '12px Inter, sans-serif';
        const lines = ann.note.split('\n');
        const lineH = 16;
        ctx.fillStyle = (ann.color || ANNOTATION_COLORS[0]) + '15';
        ctx.fillRect(x, y, 200, lines.length * lineH + 8);
        ctx.fillStyle = '#1e293b';
        lines.forEach((line: string, i: number) => {
          ctx.fillText(line, x + 4, y + 14 + i * lineH);
        });
      }
    });

    if (currentRect) {
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
      ctx.setLineDash([]);
    }
  }

  function getRelativePos(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode === 'none') return;
    const pos = getRelativePos(e);
    setIsDrawing(true);
    setStartPos(pos);
    setCurrentRect({ x: pos.x, y: pos.y, w: 0, h: 0 });
  }, [mode]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPos || mode !== 'bbox') return;
    const pos = getRelativePos(e);
    setCurrentRect({
      x: Math.min(startPos.x, pos.x),
      y: Math.min(startPos.y, pos.y),
      w: Math.abs(pos.x - startPos.x),
      h: Math.abs(pos.y - startPos.y),
    });
  }, [isDrawing, startPos, mode]);

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !currentRect || mode === 'none') return;
    setIsDrawing(false);

    if (mode === 'bbox' && currentRect.w > 5 && currentRect.h > 5) {
      const label = prompt('Label this annotation:', 'Object') || 'Annotation';
      const newAnn: Annotation = {
        id: `ann-${Date.now()}`,
        mediaId,
        type: 'bbox',
        label,
        bbox: [currentRect.x, currentRect.y, currentRect.w, currentRect.h],
        color: ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length],
        createdAt: new Date().toISOString(),
      };
      const updated = [...annotations, newAnn];
      setAnnotations(updated);
      saveAnnotations(mediaId, updated);
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 2000);
    }

    if (mode === 'note') {
      setCurrentRect(null);
      setStartPos(null);
      setNoteText('');
      return;
    }

    setCurrentRect(null);
    setStartPos(null);
  }, [isDrawing, currentRect, mode, annotations, mediaId]);

  function addNote() {
    const text = prompt('Enter note text:');
    if (!text || !canvasRef.current) return;
    const newAnn: Annotation = {
      id: `ann-${Date.now()}`,
      mediaId,
      type: 'note',
      note: text,
      bbox: [20, 20 + annotations.filter((a) => a.type === 'note').length * 60, 0, 0],
      color: ANNOTATION_COLORS[annotations.length % ANNOTATION_COLORS.length],
      createdAt: new Date().toISOString(),
    };
    const updated = [...annotations, newAnn];
    setAnnotations(updated);
    saveAnnotations(mediaId, updated);
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus('idle'), 2000);
  }

  function deleteSelected() {
    if (!selectedAnnotation) return;
    const updated = annotations.filter((a) => a.id !== selectedAnnotation);
    setAnnotations(updated);
    setSelectedAnnotation(null);
    saveAnnotations(mediaId, updated);
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus('idle'), 2000);
  }

  function downloadAnnotations() {
    const blob = new Blob([JSON.stringify(annotations, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `annotations-${mediaId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== 'none') return;
    const pos = getRelativePos(e);
    const clicked = annotations.find((ann) => {
      if (ann.type === 'bbox' && ann.bbox) {
        const [x, y, w, h] = ann.bbox;
        return pos.x >= x && pos.x <= x + w && pos.y >= y && pos.y <= y + h;
      }
      if (ann.type === 'note' && ann.bbox) {
        const [x, y] = ann.bbox;
        return pos.x >= x && pos.x <= x + 200 && pos.y >= y && pos.y <= y + 40;
      }
      return false;
    });
    setSelectedAnnotation(clicked?.id || null);
  }, [annotations, mode]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-soft">
      <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3">
        <div className="flex items-center gap-2">
          <Pencil className="h-4 w-4 text-safety-blue" />
          <h2 className="text-sm font-bold text-slate-950">Manual Annotation</h2>
        </div>
        <div className="flex items-center gap-2">
          {saveStatus === 'saving' && <LoaderCircle className="h-4 w-4 animate-spin text-blue-500" />}
          {saveStatus === 'saved' && <span className="flex items-center gap-1 text-xs text-emerald-600"><CheckCircle2 className="h-3 w-3" /> Saved</span>}
          {saveStatus === 'error' && <span className="text-xs text-red-500">Save failed</span>}
        </div>
      </div>

      <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-2">
        <button
          type="button"
          onClick={() => setMode(mode === 'bbox' ? 'none' : 'bbox')}
          className={`focus-ring inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
            mode === 'bbox' ? 'bg-safety-blue text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
          }`}
        >
          <Square className="h-3.5 w-3.5" />
          Draw Box
        </button>
        <button
          type="button"
          onClick={addNote}
          className="focus-ring inline-flex items-center gap-1.5 rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:bg-slate-200"
        >
          <Type className="h-3.5 w-3.5" />
          Add Note
        </button>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={deleteSelected}
            disabled={!selectedAnnotation}
            className="focus-ring inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-semibold text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Delete
          </button>
          <button
            type="button"
            onClick={downloadAnnotations}
            className="focus-ring inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-semibold text-slate-600 transition hover:bg-slate-100"
          >
            <Download className="h-3.5 w-3.5" />
            Export JSON
          </button>
        </div>
      </div>

      <div ref={containerRef} className="relative bg-slate-950">
        {mediaUrl ? (
          <>
            {mediaType === 'video' ? (
              <video
                src={mediaUrl}
                className="block max-h-96 w-full object-contain"
                controls
                muted
              />
            ) : (
              <img
                src={mediaUrl}
                loading="lazy"
                decoding="async"
                alt="Media for annotation"
                className="block max-h-96 w-full object-contain"
                draggable={false}
              />
            )}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 h-full w-full cursor-crosshair"
              width={containerRef.current?.querySelector('img, video')?.clientWidth || 800}
              height={containerRef.current?.querySelector('img, video')?.clientHeight || 450}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onClick={handleCanvasClick}
            />
          </>
        ) : (
          <div className="flex h-48 items-center justify-center text-sm text-slate-500">
            {mode !== 'none' ? 'Click and drag on the media below to annotate' : 'Select a frame with a preview image to annotate'}
          </div>
        )}
      </div>

      <div className="px-5 py-3 text-xs text-slate-400">
        <p>{annotations.length} annotation{annotations.length !== 1 ? 's' : ''} &middot; Data saved to localStorage &middot; <button type="button" onClick={downloadAnnotations} className="text-safety-blue hover:underline">Download JSON</button> for permanent storage</p>
      </div>
    </section>
  );
}
