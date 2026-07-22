import type { AnalysisResult } from '../types/analysis';

export interface ResultRequestToken {
  jobId: string;
  generation: number;
  controller: AbortController;
}

export class ResultRequestCoordinator {
  private generation = 0;
  private controller: AbortController | null = null;
  private selectedJobId: string | null = null;

  begin(jobId: string): ResultRequestToken {
    this.controller?.abort();
    this.generation += 1;
    this.controller = new AbortController();
    this.selectedJobId = jobId;
    return {
      jobId,
      generation: this.generation,
      controller: this.controller,
    };
  }

  cancel(): void {
    this.controller?.abort();
    this.generation += 1;
    this.controller = null;
    this.selectedJobId = null;
  }

  isCurrent(token: Pick<ResultRequestToken, 'jobId' | 'generation' | 'controller'>): boolean {
    return !token.controller.signal.aborted
      && token.generation === this.generation
      && token.jobId === this.selectedJobId;
  }

  complete(token: Pick<ResultRequestToken, 'jobId' | 'generation' | 'controller'>): void {
    if (!this.isCurrent(token)) return;
    this.controller = null;
  }
}

export function resultBelongsToJob(result: AnalysisResult, expectedJobId: string): boolean {
  if (result.jobId !== expectedJobId) return false;
  if (result.media.jobId && result.media.jobId !== expectedJobId) return false;
  if (result.frames.some((frame) => frame.jobId !== expectedJobId)) return false;
  if (result.frames.some((frame) => frame.evidenceId && !frame.evidenceId.startsWith(`${expectedJobId}:`))) return false;
  if (result.frames.some((frame) => frame.imageUrl && frame.imageUrl.includes('/api/media/') && !frame.imageUrl.includes(`/api/media/${expectedJobId}/`))) return false;
  if (result.violations?.some((finding) => finding.jobId !== expectedJobId)) return false;
  if (result.violations?.some((finding) => finding.affectedFrames.some((frame) => frame.jobId !== expectedJobId))) return false;
  if (result.events?.some((event) => event.jobId !== expectedJobId)) return false;
  if (result.events?.some((event) => event.supportingFrames.some((frame) => frame.jobId !== expectedJobId))) return false;
  return true;
}
