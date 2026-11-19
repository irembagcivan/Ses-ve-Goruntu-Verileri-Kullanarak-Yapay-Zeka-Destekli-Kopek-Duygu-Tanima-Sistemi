"""Train behavior classifiers and export held-out results for a local demo."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

TARGETS = ["aktivite", "vucut", "durus", "kuyruk", "kulak"]


def emotion_rule(predictions, scores=None):
    """Illustrative rule from behavior predictions, never a verified emotion."""
    if scores:
        # Hand-authored evidence weights, not an emotion model or probabilities.
        prototypes = {
            "Happy": [("aktivite", "oynuyor_kosuyor", 3), ("kuyruk", "sallaniyor", 2), ("vucut", "gevsek", 1.5)],
            "Relaxed": [("aktivite", "hareketsiz", 3), ("vucut", "gevsek", 2), ("durus", "yatiyor", 1)],
            "Angry": [("aktivite", "havliyor_hirliyor", 3), ("vucut", "gergin", 2), ("kuyruk", "yukari", 1)],
            "Sad": [("vucut", "cekingen_kapali", 3), ("kuyruk", "asagi_bacak_arasi", 3), ("kulak", "geride_yassi", 1.5)],
        }
        evidence = {name: sum(weight * float(scores.get(target, {}).get(value, 0)) for target, value, weight in rules)
                    / sum(weight for _, _, weight in rules) for name, rules in prototypes.items()}
        ranking = sorted(evidence, key=evidence.get, reverse=True)
        best, second = ranking[:2]
        support = evidence[best]
        margin = support - evidence[second]
        clear = support >= .6 and margin >= .12
        return {"label": best if clear else "Belirsiz", "candidate": best,
                "support": "belirgin" if clear else "sinirli", "alternative": second,
                "rule_scores": evidence, "scores_are_emotion_probabilities": False,
                "reason": ("Davranış skorlarının en çok desteklediği örnek yorum. " if clear else
                           "Davranış skorlarına göre en yakın yorum; destek sınırlı veya alternatifler yakın. ")
                           + ("Çekingenlik/korku ile de uyumlu olabilir. " if best == "Sad" else "")
                           + "Kural puanları duygu olasılığı değildir."}
    if predictions.get("aktivite") == "oynuyor_kosuyor" and predictions.get("vucut") == "gevsek":
        return {"label": "Happy", "reason": "Oyun/koşma ve gevşek vücut tahmini"}
    if predictions.get("aktivite") == "havliyor_hirliyor" and predictions.get("vucut") == "gergin":
        return {"label": "Angry", "reason": "Havlama/hırlama ve gergin vücut tahmini"}
    if predictions.get("vucut") == "cekingen_kapali" and predictions.get("kuyruk") == "asagi_bacak_arasi":
        return {"label": "Sad", "reason": "Çekingen/kapalı vücut ve aşağıda kuyruk; korku da olabilir"}
    if predictions.get("vucut") == "gevsek" and predictions.get("aktivite") == "hareketsiz":
        return {"label": "Relaxed", "reason": "Gevşek vücut ve hareketsizlik tahmini"}
    return {"label": "Belirsiz", "reason": "Davranışlar tek bir duygu kuralını karşılamıyor"}


def choose_split(table):
    best, best_score = None, -1
    for train, test in GroupShuffleSplit(n_splits=200, test_size=.25, random_state=42).split(table, groups=table.source_sha256):
        score = 0
        for target in TARGETS:
            counts = table.loc[table[target] != "gorunmuyor", target].value_counts()
            for label, count in counts.items():
                # Select on class coverage only, never on model performance.
                in_train = (table.iloc[train][target] == label).sum()
                in_test = (table.iloc[test][target] == label).sum()
                score += 3 * int(in_train > 0) + int(count >= 4 and in_test > 0)
        if score > best_score:
            best, best_score = (train, test), score
    return best


def train(manifest, feature_path, output):
    source = pd.read_csv(manifest)
    features = pd.read_csv(feature_path)
    if features.video_id.duplicated().any() or source.video_id.duplicated().any():
        raise ValueError("One row per video is required")
    table = source.merge(features, on="video_id", validate="one_to_one")
    if len(table) < 12:
        raise ValueError("At least 12 processed videos required")
    train_indices, test_indices = choose_split(table)
    assert not set(table.iloc[train_indices].source_sha256) & set(table.iloc[test_indices].source_sha256)
    table["split"] = "train"
    table.loc[test_indices, "split"] = "test"
    output.mkdir(parents=True, exist_ok=False)
    table[["video_id", "source_sha256", "split"]].to_csv(output / "split.csv", index=False)
    metrics, models, samples = [], {}, {}
    for index in test_indices:
        row = table.iloc[index]
        samples[row.video_id] = {"video_id": row.video_id, "source_sha256": row.source_sha256,
                                 "pose_detection_rate": float(row.v_detection_rate),
                                 "audio_available": bool(row.a_available),
                                 "actual": {target: row[target] for target in TARGETS}, "predictions": {}, "scores": {}}
    for modality in ["visual", "audio", "fusion"]:
        columns = [c for c in features if (c.startswith("v_") and modality != "audio") or (c.startswith("a_") and modality != "visual")]
        x = table[columns].replace([np.inf, -np.inf], np.nan)
        models[modality] = {}
        for target in TARGETS:
            training = [i for i in train_indices if table.iloc[i][target] != "gorunmuyor"]
            testing = [i for i in test_indices if table.iloc[i][target] != "gorunmuyor"]
            if not training or not testing:
                continue
            classifier = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_depth=8,
                                                              class_weight="balanced", random_state=42, n_jobs=4))
            classifier.fit(x.iloc[training], table.iloc[training][target])
            predicted = classifier.predict(x.iloc[testing])
            actual = table.iloc[testing][target]
            labels = sorted(set(table.loc[table[target] != "gorunmuyor", target]))
            dummy = DummyClassifier(strategy="most_frequent").fit(x.iloc[training], table.iloc[training][target])
            majority = dummy.predict(x.iloc[testing])
            metric = {"modality": modality, "target": target, "train_videos": len(training), "test_videos": len(testing),
                      "accuracy": accuracy_score(actual, predicted), "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                      "macro_f1": f1_score(actual, predicted, labels=labels, average="macro", zero_division=0),
                      "majority_accuracy": accuracy_score(actual, majority),
                      "majority_macro_f1": f1_score(actual, majority, labels=labels, average="macro", zero_division=0),
                      "classes": labels, "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist()}
            metrics.append(metric)
            models[modality][target] = {"model": classifier, "columns": columns}
            test_predictions = classifier.predict(x.iloc[test_indices])
            probabilities = classifier.predict_proba(x.iloc[test_indices])
            for index, prediction, scores in zip(test_indices, test_predictions, probabilities):
                sample = samples[table.iloc[index].video_id]
                sample["predictions"].setdefault(modality, {})[target] = prediction
                sample["scores"].setdefault(modality, {})[target] = dict(zip(classifier.classes_, map(float, scores)))
            print(f"{modality:6} {target:8} accuracy={metric['accuracy']:.3f} macroF1={metric['macro_f1']:.3f}", flush=True)
    for sample in samples.values():
        sample["emotion_rule"] = {m: emotion_rule(p, sample["scores"][m]) for m, p in sample["predictions"].items()}
    report = {"schema_version": 1, "videos": len(table), "train_videos": len(train_indices), "test_videos": len(test_indices),
              "evaluation": "fixed video-level holdout; exact duplicate content grouped by SHA256",
              "limitations": ["Köpek kimliği bilinmiyor; aynı köpek iki bölümde bulunabilir.",
                              "Duygu gösterimi davranış tahminlerinden türetilen örnek kurallardır; gerçek duygu etiketi yoktur.",
                              "Ses tüm ortamı içerir; yalnız köpeğin sesi ayrıştırılmamıştır.",
                              "Her videonun ilk 60 saniyesi, saniyede yaklaşık 2 kare ile incelenmiştir.",
                              "Tek ve küçük test ayrımıdır; sonuçlar genelleme garantisi vermez."],
              "metrics": metrics, "samples": list(samples.values()), "scores_are_calibrated": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pd.DataFrame([{k: v for k, v in m.items() if k not in {"classes", "confusion_matrix"}} for m in metrics]).to_csv(output / "metrics.csv", index=False)
    signature = json.loads((feature_path.parent / f"{table.iloc[0].source_sha256}.json").read_text())["signature"]
    joblib.dump({"models": models, "targets": TARGETS, "fit_split": "train_only", "extraction_signature": signature}, output / "models.joblib")
    print(f"Report saved: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.manifest, args.features, args.output)
