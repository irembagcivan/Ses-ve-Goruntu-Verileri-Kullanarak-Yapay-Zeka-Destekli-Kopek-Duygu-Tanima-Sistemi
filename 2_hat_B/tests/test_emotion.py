import json

import pandas as pd
import pytest

pytest.importorskip("sklearn")
from src.emotion.run import features, train, predict


def test_missing_motion_is_not_zero(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"global_mean_speed_normalized_s": [None, None]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No measurable"):
        features(path)


def test_grouped_training_and_prediction(tmp_path, capsys):
    rows = []
    for i in range(12):
        motion = tmp_path / f"{i}.csv"
        pd.DataFrame({"global_mean_speed_normalized_s": [i, i + 1, i + 2]}).to_csv(motion, index=False)
        rows.append({"video_id": str(i), "group_id": str(i // 2),
                     "label": "Happy" if i < 6 else "Sad", "motion_csv": motion.name})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    output = tmp_path / "trained"
    train(manifest, output)
    validation = pd.read_csv(output / "validation_predictions.csv")
    assert validation.groupby("group_id").fold.nunique().max() == 1
    assert json.loads((output / "metrics.json").read_text())["videos"] == 12
    capsys.readouterr()
    predict(output / "model.joblib", motion)
    assert json.loads(capsys.readouterr().out)["prediction"] in {"Happy", "Sad"}
