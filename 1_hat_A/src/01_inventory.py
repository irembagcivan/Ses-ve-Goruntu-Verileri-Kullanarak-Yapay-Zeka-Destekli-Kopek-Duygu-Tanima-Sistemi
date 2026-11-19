"""
Aşama 2a - Video envanteri
--------------------------
Amaç : Klasördeki tüm videoların teknik özelliklerini (süre, çözünürlük, fps, ses var mı,
       ses seviyesi, yinelenen dosya) tek bir tabloya (CSV) dökmek.
Neden: Etiketlemeye başlamadan önce veriyi tanımalıyız. Hangi videolarda ses var,
       süreler ne kadar, bozuk veya yinelenen dosya var mı sorularına sayıyla cevap veririz.

Çalıştırma (VS Code terminalinde, proje klasöründe, sanal ortam açıkken):
    python src/01_inventory.py --data-dir "/Users/KULLANICI/Desktop/Internship dataset"

Çıktılar:
    outputs/inventory.csv           -> her video için bir satır
    outputs/inventory_summary.txt   -> özet istatistikler (bize bunu gönderin)
"""
import argparse
import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def list_files(data_dir: Path):
    """Videoları ve atlanan dosyaları ayırır. macOS'un gizli '._' ve .DS_Store dosyaları atlanır."""
    videos, skipped = [], []
    for p in sorted(data_dir.rglob("*")):
        if p.is_dir():
            continue
        if p.name.startswith("._") or p.name == ".DS_Store" or "__MACOSX" in p.parts:
            skipped.append((p, "macOS sistem dosyası"))
        elif p.suffix.lower() in VIDEO_EXT:
            videos.append(p)
        else:
            skipped.append((p, "video değil"))
    return videos, skipped


def run_ffprobe(path: Path):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def parse_rotation(vstream):
    """iPhone videoları genelde yan kaydedilir ve 'rotation' bilgisiyle döndürülür."""
    for sd in vstream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"])))
    rot = (vstream.get("tags") or {}).get("rotate")
    return int(rot) if rot else 0


def parse_probe(info):
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})
    tags = {k.lower(): val for k, val in (fmt.get("tags") or {}).items()}
    out = {"duration_s": float(fmt["duration"]) if fmt.get("duration") else np.nan,
           "creation_time": tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time"),
           "has_video": v is not None, "has_audio": a is not None}
    if v:
        rot = parse_rotation(v)
        w, h = int(v["width"]), int(v["height"])
        if abs(rot) in (90, 270):          # ekranda görünen boyut
            w, h = h, w
        try:
            fps = float(Fraction(v.get("avg_frame_rate", "0/1")))
        except (ZeroDivisionError, ValueError):
            fps = np.nan
        out.update(width=w, height=h, orientation="dikey" if h > w else "yatay",
                   rotation=rot, fps=round(fps, 2), video_codec=v.get("codec_name"))
    if a:
        out.update(audio_codec=a.get("codec_name"),
                   audio_sr=int(a["sample_rate"]) if a.get("sample_rate") else np.nan)
    return out


def audio_stats(path: Path, sr=16000, frame_ms=50):
    """Sesi 16 kHz mono'ya çevirip 50 ms'lik çerçevelerde ses şiddetini (dB) hesaplar.
    NOT: Bu havlama tespiti DEĞİLDİR; sadece 'ses düz mü, hareketli mi' ipucu verir."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) == 0:
        return {}
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(sr * frame_ms / 1000)
    k = len(x) // n
    if k < 2:
        return {}
    frames = x[: k * n].reshape(k, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    floor = np.percentile(db, 10)
    p95 = np.percentile(db, 95)
    return {"audio_mean_db": round(float(db.mean()), 1),
            "audio_max_db": round(float(db.max()), 1),
            "audio_floor_db": round(float(floor), 1),
            "audio_dyn_range_db": round(float(p95 - floor), 1),
            "audio_active_ratio": round(float((db > floor + 15).mean()), 3)}


def md5_of(path: Path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def name_parts(stem: str):
    m = re.match(r"^(.*?)(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos, skipped = list_files(args.data_dir)
    print(f"{len(videos)} video bulundu, {len(skipped)} dosya atlandı.")

    rows = []
    for p in tqdm(videos, desc="Envanter"):
        row = {"file": p.name, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 1)}
        prefix, num = name_parts(p.stem)
        row.update(name_prefix=prefix, name_number=num)
        info = run_ffprobe(p)
        if info is None:
            row["error"] = "ffprobe okuyamadı (bozuk dosya olabilir)"
            rows.append(row)
            continue
        row.update(parse_probe(info))
        if row.get("has_audio"):
            row.update(audio_stats(p))
        row["md5"] = md5_of(p)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["name_prefix", "name_number"]).reset_index(drop=True)
    # Sesi düz görünen videolar: sadece dinleyerek kontrol sırasını belirlemek için işaret
    if "audio_dyn_range_db" in df:
        df["audio_flat_flag"] = df["has_audio"] & (df["audio_dyn_range_db"] < 10)
    df.to_csv(args.out_dir / "inventory.csv", index=False)

    # ---- Özet ----
    L = []
    L.append(f"Klasör: {args.data_dir}")
    L.append(f"Video sayısı: {len(df)}   |   Atlanan dosya: {len(skipped)}")
    for p, why in skipped:
        L.append(f"   atlandı: {p.name}  ({why})")
    L.append(f"Toplam boyut: {df.size_mb.sum() / 1000:.2f} GB")
    if "error" in df:
        bad = df[df["error"].notna()]
        L.append(f"Okunamayan dosya: {len(bad)} -> {list(bad.file)}")
    ok = df[df.get('has_video', False) == True] if "has_video" in df else df.iloc[0:0]
    if len(ok):
        d = ok.duration_s
        L.append(f"Toplam süre: {d.sum() / 60:.1f} dk | ortalama {d.mean():.1f} sn | "
                 f"medyan {d.median():.1f} | min {d.min():.1f} | max {d.max():.1f}")
        L.append(f"Çözünürlük (görünen):\n{ok.assign(res=ok.width.astype(int).astype(str) + 'x' + ok.height.astype(int).astype(str)).res.value_counts().to_string()}")
        L.append(f"Yönelim:\n{ok.orientation.value_counts().to_string()}")
        L.append(f"FPS:\n{ok.fps.value_counts().to_string()}")
        L.append(f"Video codec:\n{ok.video_codec.value_counts().to_string()}")
        L.append(f"Ses kanalı var mı:\n{ok.has_audio.value_counts().to_string()}")
        if "audio_flat_flag" in ok:
            flat = ok[ok.audio_flat_flag == True]
            L.append(f"Sesi düz görünen (dinleyerek kontrol edilecek): {len(flat)} -> {list(flat.file)}")
        L.append(f"Dosya adı gruplarına göre sayı:\n{ok.name_prefix.value_counts().to_string()}")
        if ok.creation_time.notna().any():
            L.append(f"Çekim zamanı (metadata) aralığı: {ok.creation_time.dropna().min()} -> {ok.creation_time.dropna().max()}")
        else:
            L.append("Çekim zamanı metadata'sı bulunamadı.")
    dup = df[df.md5.notna() & df.duplicated("md5", keep=False)] if "md5" in df else df.iloc[0:0]
    L.append(f"Birebir aynı (yinelenen) dosya: {len(dup)}" +
             (f" -> {dup.sort_values('md5')[['file', 'md5']].to_string(index=False)}" if len(dup) else ""))
    text = "\n".join(L)
    print("\n" + text)
    (args.out_dir / "inventory_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nKaydedildi: {args.out_dir / 'inventory.csv'} ve inventory_summary.txt")


if __name__ == "__main__":
    main()
