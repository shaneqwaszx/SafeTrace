# Annotation Guide

## Unit of annotation

Annotate a continuous event interval in the original video. Use the first frame
where the evidence is reasonably visible as `startSeconds` and the last such
frame as `endSeconds`. Do not label each sampled frame as a separate incident.

## Procedure

1. Watch the full clip once without pausing.
2. Review candidate intervals frame by frame.
3. Select the profile and one canonical expected finding.
4. Mark `positive`, `negative`, or `uncertain`.
5. Record visibility limitations and confidence.
6. A second reviewer independently checks every positive and uncertain row.

## Ambiguity rules

- Glare, blur, darkness, camera angle, and occlusion that prevent confirmation
  must be `insufficient_visibility` / `uncertain`, not a violation.
- For a partially visible belt-like path, mark uncertain unless continuity
  across the torso is clear.
- In multi-person scenes, identify the relevant occupant/person in notes and
  use box or mask references when available.
- Record simultaneous violations as separate rows sharing the same media.
- A reviewer disagreement remains pending until resolved; it is excluded from
  accuracy scoring.
- Default timestamp matching tolerance is two seconds unless the manifest
  declares another value.

Do not use SafeTrace predictions, annotated overlays, or VLM text to create the
human label. Review the original media.
