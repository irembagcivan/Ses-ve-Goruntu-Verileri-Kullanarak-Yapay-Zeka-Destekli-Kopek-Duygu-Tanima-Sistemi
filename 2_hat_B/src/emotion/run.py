"""Train and predict: python -m src.emotion.run --help."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

CLASSES = ["Happy", "Sad", "Angry", "Relaxed"]


def features(path):
    frame = pd.read_csv(path)
    # Use normalized movement only; filenames, labels and absolute pixel sizes
    # must never become input features. Failed/missing observations stay missing.
    columns = sorted(c for c in frame if "normalized" in c)
    if not columns or frame.empty:
        raise ValueError(f"No motion observations: {path}")
    result = {}
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for stat, value in {
            "median": values.median(), "mean": values.mean(),
            "std": values.std(), "p90": values.quantile(.9),
            "missing": values.isna().mean(),
        }.items():
            result[f"{name}__{stat}"] = float(value)
    movement = [v for k, v in result.items() if not k.endswith("__missing")]
    if not any(np.isfinite(v) for v in movement):
        raise ValueError(f"No measurable motion: {path}")
    return result


def estimator():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=2, random_state=42, n_jobs=-1),
    )


def train(manifest, output):
    table = pd.read_csv(manifest, dtype=str).fillna("")
    required = {"video_id", "group_id", "label", "motion_csv"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError(f"Manifest requires nonempty columns: {sorted(required)}")
    if table[list(required)].eq("").any().any():
        raise ValueError("Manifest fields cannot be empty")
    if table.video_id.duplicated().any():
        raise ValueError("Duplicate video_id: one row per original video required")
    if not set(table.label).issubset(CLASSES) or table.label.nunique() < 2:
        raise ValueError(f"At least two labels from {CLASSES} required")
    paths = [(manifest.parent / p).resolve() for p in table.motion_csv]
    if len(set(paths)) != len(paths):
        raise ValueError("The same motion file cannot appear twice")
    x = pd.DataFrame([features(p) for p in paths])
    y, groups = table.label, table.group_id
    folds = min(5, int(table.groupby("label").group_id.nunique().min()))
    if folds < 2:
        raise ValueError("Need at least two independent groups per class for evaluation")
    prediction = np.empty(len(y), dtype=object)
    fold_ids = np.zeros(len(y), dtype=int)
    for i, (train_idx, test_idx) in enumerate(
        StratifiedGroupKFold(folds, shuffle=True, random_state=42).split(x, y, groups), 1
    ):
        if set(y.iloc[train_idx]) != set(y):
            raise ValueError("A training fold lacks a class; add independent videos/groups")
        model = estimator().fit(x.iloc[train_idx], y.iloc[train_idx])
        prediction[test_idx] = model.predict(x.iloc[test_idx])
        fold_ids[test_idx] = i
    output.mkdir(parents=True, exist_ok=False)
    labels = sorted(set(y))
    report = {
        "model": "pose_motion_random_forest", "videos": len(y), "folds": folds,
        "evaluation": "out_of_fold_grouped_by_group_id",
        "macro_f1": f1_score(y, prediction, average="macro"),
        "classification_report": classification_report(y, prediction, output_dict=True, zero_division=0),
        "classes": labels, "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
        "limitations": ["Pose/motion only; audio not used.",
                        "Group IDs must identify original source/dog, not extracted clips.",
                        "Scores measure supplied dataset labels, not verified animal feelings."],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table.assign(predicted=prediction, fold=fold_ids).to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(report["confusion_matrix"], index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
    joblib.dump({"model": estimator().fit(x, y), "columns": list(x), "classes": labels}, output / "model.joblib")
    print(json.dumps({"macro_f1": report["macro_f1"], "videos": len(y), "output": str(output)}))


def predict(model_path, motion):
    # Load only a model produced locally by this command (joblib is executable).
    bundle = joblib.load(model_path)
    x = pd.DataFrame([features(motion)])
    if set(x) != set(bundle["columns"]):
        raise ValueError("Motion feature schema differs from training")
    probabilities = bundle["model"].predict_proba(x[bundle["columns"]])[0]
    result = dict(zip(bundle["model"].classes_, map(float, probabilities)))
    print(json.dumps({"prediction": max(result, key=result.get), "model_scores": result,
                      "modality": "pose_motion", "scores_are_calibrated": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--motion", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.manifest.resolve(), args.output)
    else:
        predict(args.model, args.motion)


if __name__ == "__main__":
    main()
