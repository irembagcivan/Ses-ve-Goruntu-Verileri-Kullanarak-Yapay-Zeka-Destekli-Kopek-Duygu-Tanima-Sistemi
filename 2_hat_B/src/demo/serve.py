"""Serve the private local behavior demo; bind to localhost only."""
import argparse
import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


def handler(root):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self):
            if self.path.split("?")[0] == "/":
                self.path = "/index.html"
            super().do_GET()

        def log_message(self, format, *args):
            pass
    return Handler


def export_static(results, embed_media=True, filename="index.html", clip_directory="clips"):
    report = json.loads((results / "report.json").read_text())
    public = results / "public"
    public.mkdir(exist_ok=True)
    if Path(filename).name != filename or not filename.endswith(".html"):
        raise ValueError("Output filename must be a plain .html filename")
    if Path(clip_directory).name != clip_directory:
        raise ValueError("Clip directory must be a plain directory name")
    # A downloaded HTML loses its relative clips/ folder. Inline the verified
    # previews so opening or moving this single file cannot break video paths.
    missing = []
    for sample in report.get("samples", []):
        clip = public / clip_directory / f"{sample['source_sha256']}.mp4"
        sample["video_url"] = f"{clip_directory}/{sample['source_sha256']}.mp4"
        if clip_directory in {"clips_v3","clips_v4"}:
            metadata = clip.with_suffix(".json")
            if metadata.is_file():
                sample["preview_metadata"] = json.loads(metadata.read_text())
        if embed_media:
            if not clip.is_file():
                missing.append(sample["video_id"])
            else:
                sample["video_data"] = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode("ascii")
    if missing:
        raise FileNotFoundError(f"Missing previews: {', '.join(missing)}")
    data = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    html = (Path(__file__).parent / "index.html").read_text()
    html = html.replace("<script>", '<script id="demo-data" type="application/json">' + data + '</script><script>', 1)
    (public / filename).write_text(html)
    (public / "report.json").write_text((results / "report.json").read_text())
    return public


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static", action="store_true", help="Export a page that opens directly without a server")
    parser.add_argument("--filename", default="index.html")
    parser.add_argument("--clip-directory", default="clips")
    args = parser.parse_args()
    if not (args.results / "report.json").is_file():
        parser.error("Training report not found")
    # Serve only exported presentation assets; never expose model.joblib or local paths.
    public = export_static(args.results, embed_media=args.static, filename=args.filename, clip_directory=args.clip_directory)
    if args.static:
        print(f"Demo file: {(public / args.filename).resolve()}")
    else:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(public))
        print(f"Demo: http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
