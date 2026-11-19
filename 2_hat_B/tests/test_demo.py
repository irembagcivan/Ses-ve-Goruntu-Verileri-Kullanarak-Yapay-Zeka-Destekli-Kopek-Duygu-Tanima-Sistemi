import csv
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from src.demo.extract import aggregate, clean_json
from src.demo.train import choose_split, emotion_rule
from src.demo.prepare import prepare
from src.demo.serve import export_static


def test_feature_summary_retains_missing_and_ignores_nonfinite():
    result = aggregate([[1, np.nan], [3, np.inf]], "v")
    assert result["v_0_mean"] == 2
    assert np.isnan(result["v_1_mean"])
    assert clean_json(result)["v_1_mean"] is None
    json.dumps(clean_json(result), allow_nan=False)


def test_exact_duplicate_videos_never_cross_split():
    table = pd.DataFrame([{"source_sha256": str(i//2), "aktivite": str(i%3),
                           "vucut": str(i%2), "durus": "ayakta", "kuyruk": "notr", "kulak": "notr"}
                          for i in range(40)])
    training, testing = choose_split(table)
    assert set(table.iloc[training].source_sha256).isdisjoint(table.iloc[testing].source_sha256)


def test_emotion_rule_leaves_ambiguous_behaviors_unresolved():
    assert emotion_rule({"aktivite": "yuruyor", "vucut": "gergin"})["label"] == "Belirsiz"
    assert emotion_rule({"aktivite": "oynuyor_kosuyor", "vucut": "gevsek"})["label"] == "Happy"


def test_prepare_excludes_incomplete_labels_and_resolves_known_rename(tmp_path, monkeypatch):
    import src.demo.prepare as module
    (tmp_path / "IMG_5417 kopyası.MOV").write_bytes(b"video content")
    monkeypatch.setattr(module, "workbook_rows", lambda _: [
        {"A": "IMG_5417.MOV", "B": "notr", "C": "notr", "D": "ayakta", "E": "yuruyor", "F": "gevsek"},
        {"A": "missing.mov", "B": "notr"},
    ])
    workbook = tmp_path / "labels.xlsx"
    workbook.write_bytes(b"mock workbook")
    prepare(workbook, tmp_path, tmp_path / "prepared")
    with (tmp_path / "prepared/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["video_id"] == "IMG_5417.MOV"
    assert rows[0]["video_path"].endswith("IMG_5417 kopyası.MOV")
    assert json.loads((tmp_path / "prepared/audit.json").read_text())["included"] == 1


def test_static_demo_embeds_report_without_executable_label_markup(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"label": "</script><script>alert(1)</script>"}))
    root = export_static(tmp_path)
    page = (root / "index.html").read_text()
    assert 'id="demo-data"' in page
    assert "</script><script>alert(1)</script>" not in page
    assert "Promise.resolve(JSON.parse(embedded.textContent))" in page


def test_portable_demo_contains_exact_video_bytes(tmp_path):
    import base64
    root = tmp_path / "public/clips"
    root.mkdir(parents=True)
    (root / "abc.mp4").write_bytes(b"example video bytes")
    (tmp_path / "report.json").write_text(json.dumps({"samples": [{"source_sha256": "abc", "video_id": "a.mov"}]}))
    export_static(tmp_path)
    page = (tmp_path / "public/index.html").read_text()
    payload = page.split('type="application/json">', 1)[1].split('</script>', 1)[0]
    data = json.loads(payload)
    assert base64.b64decode(data['samples'][0]['video_data'].split(',', 1)[1]) == b"example video bytes"
    assert "video_data" not in (tmp_path / "public/report.json").read_text()


def test_weak_rule_evidence_is_not_reported_as_certain_emotion():
    scores = {"aktivite": {"yuruyor": .4, "oynuyor_kosuyor": .2, "hareketsiz": .2, "havliyor_hirliyor": .2},
              "vucut": {"gevsek": .34, "gergin": .33, "cekingen_kapali": .33}}
    result = emotion_rule({}, scores)
    assert result["label"] == "Belirsiz"
    assert result["candidate"] in {"Happy", "Relaxed", "Angry", "Sad"}
    assert result["support"] == "sinirli"
    assert not result["scores_are_emotion_probabilities"]


def test_server_export_points_to_selected_preview_version(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"samples": [{"source_sha256": "abc", "video_id": "a.mov"}]}))
    export_static(tmp_path, embed_media=False, clip_directory="clips_v3")
    page = (tmp_path / "public/index.html").read_text()
    payload = page.split('type="application/json">',1)[1].split('</script>',1)[0]
    assert json.loads(payload)["samples"][0]["video_url"] == "clips_v3/abc.mp4"
