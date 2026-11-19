# Human review of the 30 incomplete videos

The next data improvement is completing genuine behavior annotations. The review
workspace shows only the original 30 `unlabelled` videos. Existing 72 train and
25 test rows are not editable. Human fields preserve existing partial annotations
but never start from model predictions. Suggestions are separate, collapsed by
default, and close when switching video. No accept-all-suggestions action exists.

Priority is a heuristic: half mean normalized predictive entropy and half mean
rarity of the CV-selected predicted classes in the original training set.
Rarity is max(0,1 - training_count/10). This is not a validated error detector,
calibrated uncertainty, or evidence that any proposed rare class is actually present.
Labels are video-level behavior observations, not emotion labels. If multiple
dogs differ or behavior is unclear, leave the record blocked and explain in notes.

## Build

```sh
python -m src.demo.annotation_review build \
  --manifest results/demo_v4/manifest.csv --report results/demo_v7/report.json \
  --clips results/demo_v4/public/clips_v4 \
  --output results/annotation_review_v1/etiket_inceleme.html
```

The standalone HTML embeds all 30 existing skeletal previews. For full recordings,
select the source video folder and original mode. The folder remains local; no
video is uploaded. Review the full recording, fill all five behavior labels and
choose the full-recording review status only when actually reviewed. Short-preview
and blocked records are retained as drafts, not imported as completed annotations.

Progress is stored locally when the browser allows it. Explicitly download the
JSON backup; it can be reimported into the same manifest version. The backup has
only human label fields, review status, notes and source IDs, not model suggestions
or video data. A failed validation does not partially import records.

## Validated import

```sh
python -m src.demo.annotation_review apply \
  --manifest results/demo_v4/manifest.csv \
  --review /path/to/canine_insan_etiketleri.json \
  --output results/human_review_v1
```

Import verifies source SHA256, a canonical manifest fingerprint, allowed behavior
values, completion and explicit human review status. It rejects protected train/test
IDs, stale manifests and duplicate IDs. Outputs are a new manifest, the accepted
human-reviewed rows and provenance. The original workbook and manifest are untouched.
Original split values remain unchanged, including `unlabelled` for newly reviewed
rows. This intentionally does not retrain models or quietly change evaluation.
A later experiment can explicitly use accepted new rows as extra training data
while retaining the original 25 test videos and recording the new protocol.
