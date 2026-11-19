# YAMNet behavior comparison — 2026-11-08

Fixed existing split: 72 training, 25 held-out test and 30 incomplete videos. The latter are predicted only, never fitted. Visibility-specific targets exclude `gorunmuyor`, so the evaluated count may be below 25. Settings were fixed before evaluation; no model was selected based on these scores. This small holdout was already inspected in earlier development, so these are exploratory results, not an independent final evaluation.

All values below are macro F1. Original three models and their predictions/metrics remain unchanged.

| Target | Old audio | Event audio | Old fusion | New fusion | Dog-event fusion |
|---|---:|---:|---:|---:|---:|
| aktivite | 0.267 | 0.354 | 0.384 | 0.392 | 0.392 |
| vucut | 0.635 | 0.463 | 0.583 | 0.586 | 0.591 |
| durus | 0.347 | 0.372 | 0.307 | 0.307 | 0.307 |
| kuyruk | 0.260 | 0.237 | 0.396 | 0.494 | 0.449 |
| kulak | 0.499 | 0.525 | 0.407 | 0.416 | 0.416 |

The largest observed fusion change is tail behavior. Gains elsewhere are small or absent. Event-only audio reduces body-behavior performance relative to legacy audio; sound events alone do not reliably describe pose. Removing Speech/Music event features does not uniformly improve results. A larger dog-identity-separated evaluation is needed before generalization claims.

No temporal behavior or emotion labels are available. The emotion panel remains an illustrative behavior rule. Audio remains a mixed recording and cannot be assigned to one of several visible dogs.

Artifacts are produced locally under `results/demo_v6/`: report, event features, split, metrics and trained models. Raw source media and checkpoints are excluded from git.

Checkpoint SHA256: `13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744`.
Frozen split CSV SHA256: `37ef05e7aefd7c6080e527b24fbe7be4cbdfd87e38d2d610ca78a3dd15e165a1`.
