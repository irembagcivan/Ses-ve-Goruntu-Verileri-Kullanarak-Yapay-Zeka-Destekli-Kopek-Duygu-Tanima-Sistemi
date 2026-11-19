"""
Aşama 3a - Etiketleme için hafif video kopyaları
------------------------------------------------
Amaç : Orijinal videolar (HEVC, 1080p, 60 fps) her tarayıcıda akıcı açılmayabilir.
       Etiketleme için tarayıcıyla uyumlu (H.264, en fazla 960 px, 30 fps) kopyalar üretir.
       Ses korunur (havlamayı dinleyeceksiniz). Orijinal dosyalar DEĞİŞMEZ.
Not  : Bu kopyalar yalnızca etiketleme içindir. Model eğitiminde orijinal videoları
       kullanacağız.

Çalıştırma:
    python src/03_make_proxies.py --data-dir "/Users/KULLANICI/Desktop/Internship dataset"

Çıktı: labeling/proxies/<dosya_adı>.mp4  (ör. IMG_5354.MOV.mp4)
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def make_proxy(src: Path, dst: Path):
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-vf", "scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2",
           "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
           "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return r.stderr.strip()[-300:] or "bilinmeyen hata"
    tmp.replace(dst)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--inventory", default=Path("outputs/inventory.csv"), type=Path)
    ap.add_argument("--out-dir", default=Path("labeling"), type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.inventory.exists():
        raise SystemExit("outputs/inventory.csv yok. Önce 01_inventory.py çalıştırın.")
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["duration_s"].notna()]
    proxy_dir = args.out_dir / "proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for _, r in tqdm(inv.iterrows(), total=len(inv), desc="Kopyalar"):
        dst = proxy_dir / f"{r.file}.mp4"
        if dst.exists() and not args.force:
            continue
        err = make_proxy(Path(r.path), dst)
        if err:
            failed.append((r.file, err))
    print(f"\nTamamlandı: {len(list(proxy_dir.glob('*.mp4')))} kopya -> {proxy_dir}")
    for name, err in failed:
        print(f"HATA {name}: {err}")


if __name__ == "__main__":
    main()
