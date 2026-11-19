"""
Aşama 2b - Kontak sayfaları
---------------------------
Amaç : Her videodan eşit aralıklı N kare çıkarıp tek bir görsele (kontak sayfası) dizmek.
Neden: (1) Videoların içeriğini hızlıca gözden geçirmek (köpek görünüyor mu, kuyruk/kulak
       görünüyor mu, kaç köpek var, görüntü kalitesi nasıl?)
       (2) Etiketleme şemasını gerçekten görünen şeylere göre belirlemek.
Bu sayfalar etiketleme için yardımcıdır; etiketi bunlara bakarak değil, videoyu izleyerek koyacağız.

Çalıştırma (önce 01_inventory.py çalışmış olmalı):
    python src/02_contact_sheets.py --data-dir "/Users/KULLANICI/Desktop/Internship dataset"
Hızlı deneme için: --limit 5

Çıktı: outputs/contact_sheets/<video_adı>.jpg
"""
import argparse
import io
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

TITLE_H = 30


def grab_frame(path: Path, t: float, height: int):
    """ffmpeg ile t. saniyedeki kareyi alır (iPhone döndürme bilgisini ffmpeg otomatik uygular)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", f"scale=-2:{height}", "-f", "image2pipe", "-vcodec", "png", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return Image.open(io.BytesIO(r.stdout)).convert("RGB")


def get_font():
    try:
        return ImageFont.load_default(size=16)
    except TypeError:                       # eski Pillow
        return ImageFont.load_default()


def make_sheet(path: Path, duration: float, title: str, n_frames: int, cols: int, height: int):
    times = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    frames = [grab_frame(path, t, height) for t in times]
    ref = next((f for f in frames if f is not None), None)
    if ref is None:
        return None
    w, h = ref.size
    rows = -(-n_frames // cols)
    sheet = Image.new("RGB", (cols * w, TITLE_H + rows * h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    font = get_font()
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    for i, (f, t) in enumerate(zip(frames, times)):
        x, y = (i % cols) * w, TITLE_H + (i // cols) * h
        if f is None:
            f = Image.new("RGB", (w, h), (90, 90, 90))
        sheet.paste(f.resize((w, h)), (x, y))
        draw.text((x + 6, y + 4), f"{t:.1f}s", fill=(255, 255, 0), font=font)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--limit", type=int, default=None, help="sadece ilk N video (deneme için)")
    ap.add_argument("--force", action="store_true", help="var olan sayfaların üzerine yaz")
    args = ap.parse_args()

    inv_path = args.out_dir / "inventory.csv"
    if not inv_path.exists():
        raise SystemExit("outputs/inventory.csv yok. Önce 01_inventory.py çalıştırın.")
    inv = pd.read_csv(inv_path)
    inv = inv[inv["duration_s"].notna()]
    if args.limit:
        inv = inv.head(args.limit)

    sheet_dir = args.out_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for _, r in tqdm(inv.iterrows(), total=len(inv), desc="Kontak sayfaları"):
        out = sheet_dir / f"{Path(r.file).stem}.jpg"
        if out.exists() and not args.force:
            continue
        audio = "ses var" if r.has_audio else "SES YOK"
        title = f"{r.file} | {r.duration_s:.1f} sn | {int(r.width)}x{int(r.height)} | {audio}"
        sheet = make_sheet(Path(r.path), float(r.duration_s), title,
                           args.n_frames, args.cols, args.height)
        if sheet is None:
            failed.append(r.file)
            continue
        sheet.save(out, quality=85)
    print(f"\nTamamlandı. Sayfalar: {sheet_dir}")
    if failed:
        print(f"Kare alınamayan videolar ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
