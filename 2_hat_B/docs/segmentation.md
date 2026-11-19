# Dog instance segmentation

A protocol placeholder existed previously, but no working segmenter. The release
now implements `YoloDogSegmenter` and exports actual per-instance binary masks.
It loads an explicit local YOLOv8 segmentation checkpoint, validates the Segment
architecture/task and dog-class metadata, and uses `retina_masks=True` to obtain
input-image coordinates. Wrong-size masks fail explicitly rather than stretching
letterbox pixels. Non-dog classes, nonfinite/empty masks and low confidence are
excluded. Missing segmentation output never becomes a rectangular box mask.

The default is YOLOv8s-seg with confidence .30. Every displayed frame receives
fresh mask inference, at up to 8fps within the first 60 seconds. The long image
edge is at most 960 pixels. Mask PNGs are binary 0/255 at the exported video size.
Metadata retains source dimensions, exported dimensions, source frame indices,
timestamps, confidence, area, frame-local IDs and checkpoint SHA256. Separate
instances are not collapsed into a single semantic object. The preview mask is
a union for display only; the ZIP keeps each instance independently.

The analyzer produces segmented.mp4, overlay.jpg, mask.png, instance_masks.zip
and segmentation.json. The local HTTP API exposes only these whitelisted assets,
with byte-range playback for videos. Old stored analyses show no segmentation
panel; new uploads produce the new outputs. No automatic checkpoint download
occurs at analysis time.

```sh
python -m src.demo.segmentation --video /path/input.mov \
  --weights models/yolov8s-seg.pt --output results/my_segmentation
```

Official checkpoints:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt
Documentation: https://docs.ultralytics.com/tasks/segment/

A four-second qualitative check on the user's difficult IMG_5532 clip produced
32 output frames. The nano checkpoint found masks in 14 frames with max two dogs;
the small checkpoint found masks in 26 frames with max four dogs. These are model
output counts, not detection recall or accuracy. The small checkpoint is the demo
default for this qualitative improvement, not selected using behavior test labels.
No ground-truth masks are available, so IoU/Dice and segmentation generalization
are unmeasured. We do not claim that all visible dogs are always segmented.
Behavior training uses its original cached pose/audio features. Segmentation is
an added preprocessing/evidence output, not a retraining or a new classifier score.
