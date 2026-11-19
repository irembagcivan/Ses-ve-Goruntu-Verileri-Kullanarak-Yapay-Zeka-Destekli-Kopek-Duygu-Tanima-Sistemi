> **Tarihsel sürüm belgesi.** Beş hedef ve 72/25/30 sayıları güncel veri durumunu anlatmaz. Güncel altı hedef için [son etiket güncellemesi](labels_final_update.md), değerlendirme protokolü için [v14 denetimi](review_and_day_validation.md) kullanılmalıdır.

# CanineFusionNet: trainable multimodal behavior analysis

The project now trains a neural classifier, not just pretrained detection/pose
inference. YOLOv8n and RTMPose-S provide cached normalized pose/movement features.
YAMNet provides mean 1024-dimensional audio embeddings and sound-event summaries.
The pretrained encoders remain frozen. The fitted part is a two-branch MLP,
learned feature fusion, and five jointly trained behavior heads. This is feature-
level transfer learning, not end-to-end fine-tuning from raw pixels/waveforms.
Gallery previews separately use the newer YOLOv8s/m rescue pipeline; they do not
change the cached YOLOv8n features used for the reported experiment.

## Architecture and training

471 visual features and 1053 audio features feed separate Linear(64), LayerNorm,
GELU, Dropout(0.35), Linear(32), GELU branches. Fusion concatenates both 32-D
representations, then applies Linear(32), GELU and Dropout(0.35). Five linear
heads predict activity, body, posture, tail and ears with per-head softmax.
Fusion has 104,754 trainable parameters. The loss averages class-weighted cross
entropy across available targets, masking invisible labels. AdamW uses learning
rate .001, weight decay .01, gradient clipping 1 and seed 42. Full-batch CPU
training uses only the original 72 training videos. Thirty incomplete videos
never fit the model. The held-out 25 videos remain unchanged.

Imputation and scaling are fitted inside each training CV fold. Three-fold
GroupKFold groups exact SHA256 content, not dog identity. The training-only
candidate epochs are 20, 40, 60 and 80. Mean validation macro F1 across five
heads selects 80 epochs for visual, 80 for audio and 60 for fusion. All choices
are frozen before test evaluation. A single random seed was used. CV classes
are defined from the training-label vocabulary, including rare classes absent
from some inner folds.

## Observed results

| Neural input | Training CV mean F1 at selected epoch | Test mean F1 |
|---|---:|---:|
| Pose | 0.367 | 0.396 |
| Audio embedding/events | 0.382 | 0.324 |
| Learned fusion | 0.417 | 0.424 |
| Previous classical CV selection | Separate earlier experiment | 0.468 |

Five-target mean macro F1 is not accuracy. Fusion activity F1 is 0.519, versus
0.384 for the prior classical CV choice. Other tasks do not uniformly improve.
The 1,000-draw paired test-video bootstrap interval for fusion minus neural
visual mean F1 is [-0.076, 0.142]; for fusion minus classical selection it is
[-0.145, 0.053]. These are exploratory intervals on a small previously inspected
holdout, with no correction for model/development history. Do not claim proven
superiority or dog-identity generalization. The test has 25 distinct content
hashes; it does not guarantee distinct dogs or backgrounds.

## Run

Use the existing Python 3.11 pose environment, requirements.txt and
requirements-demo.txt. No TensorFlow installation is needed by the inference-
only YAMNet PyTorch port. Numerical parity against TensorFlow remains unmeasured;
the official synthetic smoke checks pass.

```sh
python -m src.deep_learning.train --manifest results/demo_v4/manifest.csv \
  --report results/demo_v8/report.json \
  --old-features results/demo_v1/features/features.csv \
  --new-features results/demo_v4/new_features/features.csv \
  --weights models/yamnet.h5 --output results/demo_v9
python -m src.deep_learning.temporal --report results/demo_v9/report.json \
  --model results/demo_v9/neural_fusion.pt --weights models/yamnet.h5 \
  --manifest results/demo_v4/manifest.csv --output results/demo_v10
```

Output directories must be new. For new raw videos:

```sh
python -m src.deep_learning.predict --video /path/video.mov \
  --model results/demo_v9/neural_fusion.pt --weights models/yamnet.h5 \
  --detector /path/yolov8n.pt --pose /path/superanimal_quadruped_rtmpose_s.pt \
  --output /path/new_prediction.json
```

The classifier requires both modalities. Checkpoint SHA256 checks protect the
preprocessing used in training. A raw-file smoke check on existing dog6.mov
agreed with the cached classifier predictions; it is not an unseen-dog evaluation.

Replay training without raw videos or pretrained encoders:

```sh
python -m src.deep_learning.reproduce \
  --snapshot results/demo_v9/feature_snapshot.npz \
  --metadata results/demo_v9/neural_fusion.json \
  --reference results/demo_v9/neural_fusion.pt \
  --output results/demo_v9/replayed_again.pt
```

The CPU replay reproduced every trained weight tensor exactly in the current
environment. Checkpoints contain state_dict tensors and load with weights_only.
NumPy feature snapshots load with allow_pickle=False. Sources, split, choices,
curves, metrics and the 25-video test confusion matrices are retained.

## Demonstration and limits

`results/demo_v10/public/canine_demo_v10.html` embeds all 127 previews and adds
three neural tabs, comparisons and neural/classical temporal model selection.
The neural timeline has the same 4-second windows and 2-second hop within the
first 60 seconds. These short windows differ from training distribution; no
per-window accuracy was measured. Missing pose/audio gates suppress unsupported
predictions. Multiple dogs lack verified persistent identities; mixed audio
cannot be attributed to one dog. Emotion remains a separate illustrative rule.
Appropriate emotion annotations are required for emotion training/evaluation.

The offline [12-slide defense](hoca_sunumu.html) uses actual measured curves,
ablations, confusion matrices and source-video previews. Open it in a browser,
use arrow keys, N for speaker notes and F for fullscreen. See
[speaker notes and anticipated questions](sunum_konusma_notlari.md).

Primary model references:
- https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
- https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html

## Dataset provenance

Source: Rao, Sanil and Ankitha K, *A Real World Dataset of Canine Emotions Using
Video and Audio*, Mendeley Data v2, DOI 10.17632/5g5hg9hrkv.2.
https://data.mendeley.com/datasets/5g5hg9hrkv/2
The authors describe unlabelled recordings and clustering-derived emotion classes.
This is not a substitute for verified per-video human emotion annotations. The
local source directories have no separate emotion annotation file; the project
uses the user's human behavior workbook for supervised targets.
