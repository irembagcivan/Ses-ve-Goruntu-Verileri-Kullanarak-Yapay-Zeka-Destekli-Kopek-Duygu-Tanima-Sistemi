> **Tarihsel sürüm belgesi.** Beş hedef ve 72/25/30 sayıları güncel veri durumunu anlatmaz. Güncel altı hedef için [son etiket güncellemesi](labels_final_update.md), değerlendirme protokolü için [v14 denetimi](review_and_day_validation.md) kullanılmalıdır.

# Completion audit and new-video workflow

## Start

Double-click `START_ANALYSIS.command` on this Mac, or run:

```sh
python -m src.demo.upload_server
```

Open http://127.0.0.1:8766. Choose a video and start analysis. The server binds
only to loopback. Files remain on this computer. The app accepts MOV/MP4/MKV/WEBM/AVI
up to 250 MB, analyzes at most 60 seconds, and keeps each copied upload under a
random job ID. The source file is never modified. Progress, failures, history,
seekable H.264 playback, JSON reports and four model comparisons are available.
New uploads are not added to training or the held-out test automatically.

FusionNet, temporal CNN, GRU and BiGRU use their fixed trained checkpoints.
Temporal windows in upload results use FusionNet only; model selection above
changes video-level results. Missing usable dog pose or missing audio suppresses
combined behavior output instead of filling everything and showing false confidence.
Sounds can still be shown without a visual dog. The application is dog-specific.

The image evidence uses YOLOv8m plus RTMPose on every dog detected in one selected
frame. Snapshot numbers are not track identities. Training/video-level behavior
features remain the frozen YOLOv8n pipeline and largest box per sampled frame.
The image evidence does not claim detection accuracy or persistent multi-dog analysis.

## Temporal model comparison

All three temporal networks receive aligned 1 Hz sequences up to 60 steps:
117 pose coordinates/validity-mask features and 1024 YAMNet audio features.
Each modality projects to 32 dimensions. CNN1D uses two kernel-3 temporal
convolutions and masked mean pooling. GRU uses 32 hidden units and the last valid
state. BiGRU uses two 32-unit directions and concatenates final states. All have
five behavior heads, dropout .35, AdamW lr .001 and weight decay .01, clip norm 1,
seed 42, class-weighted masked task losses. Scalers fit only valid training steps.
CNN masks padded steps after each layer; recurrent models use packed sequences.
GRU learns order, while the earlier MLP uses video-level aggregates.

Epoch candidates 10/20/40 are chosen on three grouped training CV folds before
test evaluation. All three selected 40 epochs. Original 72/25 split remains fixed;
30 incomplete videos do not fit. BiGRU is offline and uses both temporal directions.

| Model | Selected training CV mean F1 | Test mean macro F1 |
|---|---:|---:|
| CNN1D | 0.363 | 0.337 |
| GRU | 0.422 | 0.419 |
| BiGRU | 0.463 | 0.363 |
| Prior FusionNet | 0.417 | 0.424 |
| Prior classical CV choice | Separate earlier selection | 0.468 |

Temporal inputs and aggregate inputs differ, so this is an architecture-plus-input
comparison, not a controlled architecture-only ablation. Single seed, tiny
previously inspected test and unknown dog identity limit inference. BiGRU's CV/test
gap is exposed; no test-driven winner is silently substituted. No temporal behavior
labels exist. No architecture fixes the missing emotion ground truth.

```sh
python -m src.deep_learning.sequence --manifest results/demo_v4/manifest.csv \
  --report results/demo_v10/report.json --weights models/yamnet.h5 \
  --output results/demo_v11
```

## Audit

| Area | Status and evidence |
|---|---|
| New raw video via UI | Implemented: asynchronous HTTP upload, processing stages and result view |
| Raw neural inference | FusionNet and CNN/GRU/BiGRU comparisons available in upload result |
| Missing audio/pose | Explicitly suppress fused behavior, preserve available sound evidence |
| Corrupt input | Decoder validation and failed-job result; no fabricated prediction |
| Video compatibility | First-60s H.264/AAC transcode, byte-range playback |
| Upload boundaries | 250 MB, extension check, random isolated paths, bounded three active jobs |
| Local request protection | Loopback, Host/Origin checks, per-instance POST token, no external CORS |
| Sound timeline | Multilabel candidates, scores and click-to-seek; no source separation |
| Model diversity | MLP fusion, CNN1D, GRU, BiGRU plus earlier classical baselines |
| Comparison consistency | Frozen 72 train/25 test, train-only grouped CV, all results shown |
| Multi-dog image evidence | All snapshot detections can have poses, no persistent identity claim |
| Multi-dog video identity | OPEN: largest-box features can switch subject |
| Per-dog sound attribution | OPEN: mixed recording cannot be assigned to one dog |
| Other animal species | OUT OF SCOPE: trained behavior labels are canine |
| Pose accuracy benchmark | OPEN: no manually verified keypoint ground truth |
| Emotion classifier | OPEN: only behavior labels, emotion panel remains illustrative |
| Short-window accuracy | OPEN: no temporal ground truth and distribution shift |
| Fresh independent validation | OPEN: current test was inspected previously, no dog-identity split |
| Generalization/seed stability | OPEN: one seed, small and imbalanced data |

No claim is made that every project limitation is fixed. The implemented changes
close usability gaps and expand experiments; scientific limitations stay visible.
The full dataset, raw uploads, pretrained weights and trained states stay outside git.
To stop the local server, close its terminal or press Ctrl+C. Jobs persist locally
under results/live_analysis/jobs; completed job history reloads on restart. Interrupted jobs are marked failed and can be uploaded again.
