"""Read the existing annotation workbook without changing it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

TARGETS = {"B": "kuyruk", "C": "kulak", "D": "durus", "E": "aktivite", "F": "vucut"}


def workbook_rows(path):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(x.itertext()) for x in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        rows = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall(".//m:sheetData/m:row", ns):
            record = {}
            for cell in row.findall("m:c", ns):
                value, inline = cell.find("m:v", ns), cell.find("m:is", ns)
                if cell.get("t") == "s" and value is not None:
                    text = strings[int(value.text)]
                elif inline is not None:
                    text = "".join(inline.itertext())
                else:
                    text = value.text if value is not None else ""
                record["".join(c for c in cell.get("r") if c.isalpha())] = text.strip()
            rows.append(record)
    if not rows or rows[0].get("A") != "video":
        raise ValueError("Expected video annotation worksheet")
    return rows[1:]


def prepare(workbook, video_root, output):
    videos = {p.name.casefold(): p for p in video_root.iterdir() if p.suffix.lower() in {".mov", ".mp4"}}
    selected, audit = [], []
    for row in workbook_rows(workbook):
        name = row.get("A", "")
        path = videos.get(name.casefold())
        # One source video was renamed on download. Resolve only this explicit
        # known suffix, and retain the original label name in the manifest.
        if path is None:
            candidate = Path(name)
            path = videos.get(f"{candidate.stem} kopyası{candidate.suffix}".casefold())
        missing = [target for column, target in TARGETS.items() if not row.get(column)]
        if missing or path is None:
            audit.append({"video": name, "missing_labels": missing, "missing_video": path is None})
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        selected.append({"video_id": name, "video_path": str(path.resolve()), "source_sha256": digest,
                         **{target: row[col] for col, target in TARGETS.items()}, "note": row.get("G", "")})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "video_path", "source_sha256", *TARGETS.values(), "note"])
        writer.writeheader()
        writer.writerows(selected)
    report = {"included": len(selected), "excluded": audit,
              "annotation_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
              "labels_are": "human behavior annotations, not emotion ground truth",
              "dog_identity_available": False}
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"included": len(selected), "excluded": len(audit)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.workbook, args.videos, args.output)
