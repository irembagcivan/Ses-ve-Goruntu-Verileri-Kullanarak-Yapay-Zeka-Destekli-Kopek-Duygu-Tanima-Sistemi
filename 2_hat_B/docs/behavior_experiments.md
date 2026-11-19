# Training-only algorithm and modality comparison

The runner compares LogisticRegression, RandomForest and ExtraTrees with six
prespecified feature groups: pose features, legacy audio, YAMNet event summaries,
legacy fusion, fusion plus all event summaries, and fusion plus dog-event
summaries excluding Speech/Music scores. That is 18 candidates per behavior,
90 evaluations over five behaviors, 270 CV fits and 90 final train-only fits.
All algorithms use fixed parameters; this is not a test-set hyperparameter search.

## Reproduce

Use the existing pose environment and requirements-demo.txt. Generate the earlier
feature and audio reports as described in quick_demo.md and audio_events.md.

```sh
python -m src.experiments.behavior \
  --manifest results/demo_v4/manifest.csv \
  --old-features results/demo_v1/features/features.csv \
  --new-features results/demo_v4/new_features/features.csv \
  --report results/demo_v6/report.json --output results/demo_v7
python -m src.experiments.report \
  --experiment results/demo_v7/experiments.json \
  --baseline results/demo_v6/report.json --output docs/experiments_20261108.html
node tests/demo_ui_smoke.cjs src/demo/index.html results/demo_v7/report.json
```

Output directories must not already exist. To export the gallery, reuse the
existing `clips_v4` directory under the new public directory, then use
`python -m src.demo.serve --results results/demo_v7 --static --filename canine_demo_v7.html --clip-directory clips_v4`.
The clips can be copied; no pose inference needs to be repeated.

## Evaluation and selection

The existing 72-train / 25-test content-group split remains unchanged. Thirty
incomplete videos never participate in fitting. Rows with invisible targets are
excluded only for that target. Three-fold GroupKFold groups training videos by
exact content SHA256. Each candidate sees identical folds for a target. Median
imputation and, for logistic regression, scaling are learned inside each fold.
Groups identify identical content, not animals: same-dog leakage is unresolved.
Some rare classes cannot occur in every training fold; their missing-class lists
are recorded explicitly in selection.json. In particular, only one training
video has the sitting label.

All 90 CV results and five choices are saved in selection.json/cv_progress.json
before test outcomes are evaluated. Selection uses mean CV macro F1 only, with a
stable lexical tie-break; select_candidate cannot use test values. All candidates
are then refitted on training rows and evaluated for transparent comparison.
The five chosen pipelines supply a new `cv_selected` gallery option. Existing
six modes and predictions are retained.

Best CV scores have selection optimism from comparing 18 candidates. The same
25 test videos have been inspected in previous development; this is exploratory,
not a fresh independent test. Per-target paired 1,000-draw content-group bootstrap
intervals compare the selected pipeline with the old fusion pipeline using a
fixed class list; they do not account for the wider development history or
multiple comparisons. No claims of statistically established improvement follow.

The standalone [experiment report](experiments_20261108.html) includes every
candidate, input ablations, training/test differences, selected-model confusion
matrices, class counts, per-class recall and actual error examples. Five-target
mean macro F1 is an arithmetic average, not accuracy or a pooled F1 score.
Emotion labels and temporal behavior labels are unavailable; neither is evaluated.

## Observed result (8 November)

Training CV selected old fusion RF for activity/body, augmented fusion logistic
regression for posture, augmented fusion RF for tail, and event-only RF for ears.
Mean test macro F1 across five targets: 0.415 old fusion versus 0.468 CV-selected.
Activity/body stayed unchanged; posture 0.307→0.350, tail 0.396→0.494 and ears
0.407→0.525. Difference intervals touch or cross zero. This supports further
experimentation, not a claim of confirmed generalization. Event-audio predicting
ears may reflect animal/environment correlations; it does not demonstrate causality.
