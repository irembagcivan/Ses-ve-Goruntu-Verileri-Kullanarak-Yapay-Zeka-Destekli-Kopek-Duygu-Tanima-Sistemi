"""Inventory every source video and append predictions without changing test metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from src.demo.prepare import TARGETS, workbook_rows
from src.demo.train import emotion_rule


def inventory(workbook, videos, trained, output):
    annotations = workbook_rows(workbook)
    files = {p.name.casefold(): p for p in videos.iterdir() if p.is_file() and p.suffix.lower() in {".mov", ".mp4"}}
    partitions = pd.read_csv(trained / "split.csv").set_index("source_sha256").split.to_dict()
    rows, used = [], set()
    for row in annotations:
        name = row.get("A", "")
        path = files.get(name.casefold())
        if path is None:
            p = Path(name)
            path = files.get(f"{p.stem} kopyası{p.suffix}".casefold())
        if path is None:
            continue
        used.add(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append({"video_id": name, "video_path": str(path.resolve()), "source_filename": path.name,
                     "source_sha256": digest, "split": partitions.get(digest, "unlabelled"),
                     **{target: row.get(col, "") for col, target in TARGETS.items()}, "note": row.get("G", "")})
    for path in files.values():
        if path in used:
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append({"video_id": path.name, "video_path": str(path.resolve()), "source_filename": path.name,
                     "source_sha256": digest, "split": partitions.get(digest, "unlabelled"),
                     **{target: "" for target in TARGETS.values()}, "note": ""})
    table = pd.DataFrame(rows)
    if table.video_id.duplicated().any():
        raise ValueError("Duplicate video names in inventory")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "manifest.csv", index=False)
    table.loc[table.split == "unlabelled"].to_csv(output / "new_manifest.csv", index=False)
    print(json.dumps({"total": len(table), "partitions": table.split.value_counts().to_dict()}))


def predict_gallery(manifest, trained, old_features, new_features, output):
    table = pd.read_csv(manifest).fillna("")
    features = pd.concat([pd.read_csv(old_features), pd.read_csv(new_features)], ignore_index=True)
    if features.video_id.duplicated().any():
        raise ValueError("Duplicate feature rows")
    table = table.merge(features, on="video_id", validate="one_to_one")
    expected = len(pd.read_csv(manifest))
    if len(table) != expected:
        raise ValueError(f"Features missing for {expected-len(table)} videos")
    bundle = joblib.load(trained / "models.joblib")
    report = json.loads((trained / "report.json").read_text())
    report["labelled_videos"] = report["videos"]
    report["videos"] = len(table)
    samples = []
    for _, row in table.iterrows():
        samples.append({"video_id": row.video_id, "source_sha256": row.source_sha256,
                        "source_filename": row.source_filename, "split": row.split,
                        "actual": {target: row[target] or "etiket_yok" for target in TARGETS.values()},
                        "predictions": {}, "scores": {}, "emotion_rule": {}})
    for modality, targets in bundle["models"].items():
        for target, entry in targets.items():
            x = table.reindex(columns=entry["columns"])
            predictions = entry["model"].predict(x)
            scores = entry["model"].predict_proba(x)
            for sample, predicted, score in zip(samples, predictions, scores):
                sample["predictions"].setdefault(modality, {})[target] = predicted
                sample["scores"].setdefault(modality, {})[target] = dict(zip(entry["model"].classes_, map(float, score)))
        for sample in samples:
            sample["emotion_rule"][modality] = emotion_rule(sample["predictions"][modality], sample["scores"][modality])
    report["samples"] = samples
    report["gallery_partitions"] = table.split.value_counts().to_dict()
    report["limitations"].append("Galeri eğitim ve etiketsiz videoları da içerir; başarı tablosu yalnız önceki 25 test videosunundur.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Gallery predictions: {len(samples)}; test metrics unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    for name in ["workbook", "videos", "trained", "output"]:
        inv.add_argument(f"--{name}", type=Path, required=True)
    pred = sub.add_parser("predict")
    for name in ["manifest", "trained", "old-features", "new-features", "output"]:
        pred.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.workbook, args.videos, args.trained, args.output)
    else:
        predict_gallery(args.manifest, args.trained, args.old_features, args.new_features, args.output)
