"""
Aşama 5 - Özellik çıkarma (görüntü + hareket + ses)
---------------------------------------------------
Her video için üç grup sayısal özellik çıkarır ve outputs/features/<video>.npz olarak kaydeder
(yarıda kalırsa kaldığı yerden devam eder). Sonunda outputs/features_all.npz üretir.

  Görüntü  : Önceden eğitilmiş MobileNetV2 (ImageNet) ile kare başına 1280 sayılık "gömme";
             videodaki kareler üzerinde ortalama + standart sapma (2560 özellik).
             Model DONDURULMUŞTUR (eğitilmez); sadece özellik çıkarıcıdır.
  Hareket  : 12 fps, 64x64 gri karelerde ardışık kare farkı (ortalama, std, p90, maks, farkın
             tekrar sayısı) + gömmeler arası değişim. Kuyruk sallama/koşma gibi ZAMANSAL bilgi içindir.
  Ses      : 16 kHz mono sinyalden log-mel/MFCC (20) ortalama+std, spektral merkez/bant yayılımı,
             sıfır geçiş oranı, RMS enerji istatistikleri, düşük/orta/yüksek bant enerji oranları,
             "ses olayı" oranı (sessizliğin üstündeki çerçeve oranı).

Çalıştırma (Mac, .venv içinde):
    pip install torch torchvision scikit-learn matplotlib
    python src/09_extract_features.py --limit 3        # önce 3 videoyla dene
    python src/09_extract_features.py                  # hepsi (yaklaşık 10-30 dk)
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

EMB_FPS = 4          # gömme için saniyedeki kare sayısı
EMB_MAX = 48         # en fazla kare sayısı (düzgün seyrelterek)
MOT_FPS = 12
SR = 16000


def run_ffmpeg(args):
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="ignore")[:300])
    return p.stdout


def read_frames(path, fps, w, h, gray=False):
    raw = run_ffmpeg(["-i", str(path), "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo",
                      "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"])
    c = 1 if gray else 3
    n = len(raw) // (w * h * c)
    return np.frombuffer(raw[: n * w * h * c], np.uint8).reshape(n, h, w, c) if n else np.zeros((0, h, w, c), np.uint8)


def read_audio(path):
    raw = run_ffmpeg(["-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"])
    return np.frombuffer(raw, np.float32)


# ---------------- ses ----------------
def mel_filterbank(sr, n_fft, n_mels=40, fmin=20, fmax=8000):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


def dct_matrix(n_mfcc, n_mels):
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_mels)[None, :]
    m = np.cos(np.pi / n_mels * (n + 0.5) * k) * np.sqrt(2.0 / n_mels)
    m[0] *= 1 / np.sqrt(2)
    return m


AUDIO_NAMES = ([f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)] +
               ["centroid_mean", "centroid_std", "bandwidth_mean", "zcr_mean", "zcr_std",
                "rms_mean", "rms_std", "rms_max", "rms_p90", "low_ratio", "mid_ratio", "high_ratio",
                "event_ratio", "peak_to_median", "duration_s"])


def audio_features(y):
    n_fft, hop = 512, 160
    dur = len(y) / SR
    if len(y) < n_fft * 2:
        return np.full(len(AUDIO_NAMES), np.nan, np.float32)
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = np.log(spec @ mel_filterbank(SR, n_fft).T + 1e-10)
    mfcc = mel @ dct_matrix(20, 40).T
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    tot = spec.sum(axis=1) + 1e-12
    cent = (spec * freqs).sum(axis=1) / tot
    bw = np.sqrt((spec * (freqs - cent[:, None]) ** 2).sum(axis=1) / tot)
    zcr = (np.diff(np.signbit(frames), axis=1) != 0).mean(axis=1)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    e = spec.sum(axis=0)
    et = e.sum() + 1e-12
    low, mid, high = e[freqs < 500].sum() / et, e[(freqs >= 500) & (freqs < 2000)].sum() / et, e[freqs >= 2000].sum() / et
    med = np.median(rms) + 1e-8
    f = np.concatenate([mfcc.mean(0), mfcc.std(0),
                        [cent.mean(), cent.std(), bw.mean(), zcr.mean(), zcr.std(),
                         rms.mean(), rms.std(), rms.max(), np.quantile(rms, .9), low, mid, high,
                         (rms > 3 * med).mean(), rms.max() / med, dur]])
    return f.astype(np.float32)


# ---------------- hareket ----------------
MOTION_NAMES = ["mot_mean", "mot_std", "mot_p90", "mot_max", "mot_peak_rate", "mot_zero_cross",
                "emb_delta_mean", "emb_delta_std", "emb_delta_max"]


def motion_features(gray, emb):
    out = np.full(len(MOTION_NAMES), np.nan, np.float32)
    if len(gray) >= 3:
        d = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
        dc = d - d.mean()
        zc = (np.diff(np.signbit(dc)) != 0).mean() if len(dc) > 2 else 0
        peaks = ((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > d.mean() + d.std())).sum() if len(d) > 2 else 0
        dur = len(gray) / MOT_FPS
        out[:6] = [d.mean(), d.std(), np.quantile(d, .9), d.max(), peaks / max(dur, 1e-6), zc]
    if len(emb) >= 2:
        ed = np.linalg.norm(np.diff(emb, axis=0), axis=1)
        out[6:] = [ed.mean(), ed.std(), ed.max()]
    return out


# ---------------- görüntü ----------------
def load_model(pretrained):
    import torch
    import torchvision
    w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.mobilenet_v2(weights=w).features.eval()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(dev), dev


def embed(model, dev, frames):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(frames), 16):
            x = torch.from_numpy(frames[i:i + 16]).to(dev).permute(0, 3, 1, 2).float() / 255
            x = (x - mean) / std
            f = model(x).mean(dim=(2, 3))
            outs.append(f.cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1280), np.float32)


def process(path, model, dev):
    fr = read_frames(path, EMB_FPS, 224, 224)
    if len(fr) > EMB_MAX:
        fr = fr[np.linspace(0, len(fr) - 1, EMB_MAX).round().astype(int)]
    emb = embed(model, dev, fr)
    gray = read_frames(path, MOT_FPS, 64, 64, gray=True)
    try:
        aud = audio_features(read_audio(path))
    except RuntimeError:
        aud = np.full(len(AUDIO_NAMES), np.nan, np.float32)
    vis = np.concatenate([emb.mean(0), emb.std(0)]) if len(emb) else np.full(2560, np.nan, np.float32)
    return dict(vis=vis.astype(np.float32), mot=motion_features(gray, emb), aud=aud, n_frames=len(fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-dir", default=Path("labeling/proxies"), type=Path)
    ap.add_argument("--targets", default=Path("outputs/targets_and_splits.csv"), type=Path)
    ap.add_argument("--out-dir", default=Path("outputs/features"), type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="sadece yazılım testi için (ağırlık indirmez)")
    args = ap.parse_args()

    videos = pd.read_csv(args.targets, encoding="utf-8-sig").video.tolist()
    if args.limit:
        videos = videos[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, dev = load_model(not args.no_pretrained)
    print(f"Cihaz: {dev} | video: {len(videos)} | önceden eğitilmiş ağırlık: {not args.no_pretrained}")
    t0 = time.time()
    for i, v in enumerate(videos, 1):
        cache = args.out_dir / f"{v}.npz"
        if cache.exists():
            continue
        p = args.proxy_dir / f"{v}.mp4"
        if not p.exists():
            print(f"  ATLANDI (proxy yok): {v}")
            continue
        try:
            r = process(p, model, dev)
        except Exception as e:  # tek video tüm işi durdurmasın
            print(f"  HATA {v}: {e}")
            continue
        np.savez(cache, **r)
        print(f"[{i}/{len(videos)}] {v}  kare={r['n_frames']}  geçen={time.time() - t0:.0f}s")

    if args.limit:
        print("Deneme bitti (--limit). Tümünü çalıştırmak için --limit olmadan tekrar çalıştırın.")
        return
    names, V, M, A = [], [], [], []
    for v in videos:
        c = args.out_dir / f"{v}.npz"
        if c.exists():
            z = np.load(c)
            names.append(v); V.append(z["vis"]); M.append(z["mot"]); A.append(z["aud"])
    np.savez("outputs/features_all.npz", video=np.array(names), vis=np.array(V), mot=np.array(M), aud=np.array(A),
             mot_names=np.array(MOTION_NAMES), aud_names=np.array(AUDIO_NAMES))
    print(f"outputs/features_all.npz yazıldı: {len(names)} video, "
          f"vis={np.array(V).shape}, mot={np.array(M).shape}, aud={np.array(A).shape}")
    miss = sorted(set(videos) - set(names))
    if miss:
        print("EKSİK videolar (yeniden çalıştırın):", miss)


if __name__ == "__main__":
    main()
