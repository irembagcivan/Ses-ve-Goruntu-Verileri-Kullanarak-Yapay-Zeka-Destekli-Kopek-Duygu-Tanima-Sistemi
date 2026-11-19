> **Tarihsel sürüm belgesi.** Beş hedef ve 72/25/30 sayıları güncel veri durumunu anlatmaz. Güncel altı hedef için [son etiket güncellemesi](labels_final_update.md), değerlendirme protokolü için [v14 denetimi](review_and_day_validation.md) kullanılmalıdır.

# Demo v8: scope and completion

The frozen demo contract is [release_v8.json](../configs/demo/release_v8.json).
This is a demo selection, not evidence that a universally best model is established.

| Promised priority | Implemented | Validation limit |
|---|---|---|
| Add new audio information and compare speech contribution | YAMNet events, legacy audio, fusion and Speech/Music-excluded event variants | Mixed recording; not source separation. Removing event scores does not remove speech from legacy audio. |
| Show behavior and audio changes through the video | Full-recording sound events; 4-second behavior windows every 2 seconds, first 60 seconds | Experimental video-trained models applied to short windows; no temporal labels. Recordings under 1 second have no behavior windows. |
| Keep emotion interpretation defensible | Separate video-level rule panel, explicitly not a trained emotion result | Validated emotion classification is not complete; appropriate human emotion labels are absent. |
| Measure gain on the same test videos | Frozen split, 90 candidates, training-only CV selection, same 25 test videos | Previously inspected, small exploratory holdout. No fresh independent or dog-identity evaluation. |

YOLOv8s/m and RTMPose-S support detection and skeletal previews. YAMNet produces
sound-event candidates. None has a project-specific accuracy benchmark for dog
identity or sound-event timing. Detection is operational, not guaranteed correct.
Multiple dogs do not receive persistent verified identities or individually
attributed sounds. Behavior features select the largest box in each sampled frame;
that dog can change. Window quality notes expose multi-dog sampled frames.

Frozen behavior choices:

| Target | Algorithm | Input |
|---|---|---|
| Activity | Random Forest | Legacy pose + audio |
| Body | Random Forest | Legacy pose + audio |
| Posture | Logistic regression | Legacy pose + audio + YAMNet summaries |
| Tail | Random Forest | Legacy pose + audio + YAMNet summaries |
| Ears | Random Forest | YAMNet summaries |

The same frozen pipelines are applied to short windows. Shorter aggregates differ
from the video-level training distribution; scores are not calibrated probabilities.
No per-window emotion is invented. Existing video predictions and test metrics are
unchanged. The GUI exposes temporal behavior separately and explicitly labels the
emotion panel video-level.

```sh
python -m src.demo.temporal --manifest results/demo_v4/manifest.csv \
  --report results/demo_v7/report.json --model results/demo_v7/models.joblib \
  --old-features results/demo_v1/features \
  --new-features results/demo_v4/new_features --output results/demo_v8
```

Reuses cached pose sequences; no pose model rerun or model refitting. Audio spectral
statistics are recomputed from each decoded segment using the same legacy helper;
YAMNet patches fully contained in each window supply the event features. One-second
minimum window length and missing-modality gates prevent outputs without usable
input. UI chooses the latest-starting window containing playback time, and displays
no result beyond analyzed coverage. This is offline, not a causal live-stream system.

Preserve/complete human behavior annotations with annotation_review.md. Emotion
annotation needs a separate agreed protocol; behavior or sound candidates must not
be relabeled as verified emotions. New data should be used for additional training
or fresh evaluation with identities/groups, rather than repeatedly optimizing the
same 25 test videos.
