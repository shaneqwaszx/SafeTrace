# SafeTrace labelled evaluation

SafeTrace accuracy claims must come from human-labelled media listed in an
approved manifest. Historical SafeTrace result JSON is prediction output and
must never be used as ground truth.

## Workflow

1. Copy one of the templates from `evaluation/manifests/`.
2. Reference media using repository-relative or evaluation-root-relative paths.
3. Annotate event intervals using the definitions in `evaluation/annotation/`.
4. Have a second reviewer resolve uncertain or disputed labels.
5. Set `annotationStatus` to `approved` only after review.
6. Run `scripts/benchmark_safetrace_profiles.py` separately for Fast Local and
   Comprehensive Review.

Rows marked `uncertain`, `unlabelled`, or not `approved` may be used for
functional/runtime testing but are excluded from precision, recall, F1, and
the no-regression gate.

## Minimum useful dataset

For an initial internal gate, collect at least 20 approved positive events and
20 approved negative clips for each claimed profile, across at least five
vehicles/cameras and varied day/night, glare, blur, angle, and occlusion. A
release-quality claim needs a larger held-out set selected before tuning.

Keep training and held-out evaluation splits separate by source vehicle/video,
not merely by sampled frame, to avoid leakage.
