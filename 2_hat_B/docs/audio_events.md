# Full-recording audio event timeline

The gallery adds a separate YAMNet sound-event view. Existing Random Forest
behavior predictions and held-out metrics are unchanged. The event model does
not infer canine emotions or attribute audio to a particular detected dog.

Install requirements-demo.txt alongside the existing torch environment. Download
https://storage.googleapis.com/audioset/yamnet.h5 into models/yamnet.h5 (ignored).
The official architecture, features, class map and checkpoint come from
https://github.com/tensorflow/models/tree/master/research/audioset/yamnet.
This project supplies an inference-only PyTorch port with TensorFlow SAME padding
and folded batch normalization. Official silence, white-noise and sine-wave
smoke checks pass. Numerical parity against TensorFlow has not been measured.
The official speech/whistling sample also produces Speech and Whistling events.

```sh
python -m src.audio_analysis.run --manifest results/demo_v4/manifest.csv \
  --report results/demo_v4/report.json --output results/demo_v5/report.json
```

The output parent must exist. Cache keys include source SHA, checkpoint SHA and
implementation contents. The entire audio track is decoded to 16 kHz mono.
Windows span 0.975 seconds and advance by 0.48 seconds. Displayed classes are
Dog, Bark, Growling, Whimper (dog), Howl, Speech and Music. Overlapping candidates
are allowed; a score >=0.30 starts a candidate. This threshold is a demo choice,
not validated sensitivity or a calibrated probability. No temporal annotations
are available for accuracy evaluation. Unreadable/missing audio is explicit.

The waveform covers the full recording. Clicking a candidate seeks the current
video only if its timestamp exists in that source. For events beyond the short
skeletal preview, choose the original folder and original-video mode.
The full original videos stay on the user's computer.

YAMNet code and weights are licensed Apache-2.0 by the TensorFlow Authors.
See third_party/yamnet/LICENSE and the retained original source notices.

## Behavior model experiment

```sh
python -m src.audio_analysis.behavior --manifest results/demo_v4/manifest.csv \
  --old-features results/demo_v1/features/features.csv \
  --new-features results/demo_v4/new_features/features.csv \
  --report results/demo_v5/report.json --output results/demo_v6
```

This freezes the existing split and excludes all incomplete videos from fitting.
Features aggregate complete windows within the first 60 seconds, matching the
legacy extraction scope: mean, maximum, 90th percentile and fraction >=0.30 for
each displayed sound class, plus audio availability. No labels enter extraction.
Three prespecified variants use the same RF settings as before: event audio only;
legacy visual/audio plus all events; legacy visual/audio plus dog-event classes
without Speech/Music event scores. The last variant still retains legacy mixed
recording audio features, so it is not source-separated dog audio.
All variants remain visible; the test set is not used to choose a winner.
Existing three models, predictions and metrics are preserved. No per-window
behavior claims are made from video-level labels. Results are in
[audio_behavior_results.md](audio_behavior_results.md).
