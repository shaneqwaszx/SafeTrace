# SafeTrace detector candidate workflow

SafeTrace currently uses a COCO segmentation checkpoint. It can locate people
and phones, but it does not contain dedicated seatbelt, helmet, torso, head,
hand, or steering-wheel classes. Absence of those COCO labels is not proof of a
safety violation.

## Candidate scope

Train or obtain a segmentation/detection candidate with reviewed labels for at
least `person`, `seatbelt`, `helmet`, `head`, and `torso`. Add `hand`,
`cell_phone`, `driver_area`, and `steering_wheel` when the intended camera view
supports them. Split train/validation/test by vehicle or camera source so near
duplicate frames do not cross partitions.

## Promotion sequence

1. Freeze the current detector hash and Phase O benchmark configuration.
2. Audit the candidate with `scripts/audit_detector_checkpoint.py`.
3. Run the approved labelled manifest in Fast Local and Comprehensive modes.
4. Compare per-profile precision, recall, F1, false positives, event matching,
   runtime, memory, and evidence-frame counts.
5. Reject promotion if seatbelt or helmet false positives increase, F1 drops
   outside the declared tolerance, output schemas change, or soak integrity
   fails.
6. Configure a candidate with `SAFETRACE_YOLO_CKPT`; do not overwrite the
   current checkpoint. Change the default only after the no-regression gate
   passes on a representative held-out labelled set.

The template in `training/dataset.yaml.example` is intentionally path-neutral.
No model download or training is performed by the Phase O readiness pass.
