"""InfraVision rescue dashboard (Gradio). Runs on CPU.

    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # temporary public link for a friend
    python app.py --device cpu --config configs/infravision.yaml
    python app.py --colorizer infravision_tiny   # start with the old colorizer (switchable in the UI)

Single image tab: four stages (raw thermal | super-resolved thermal | AI colorized |
tactical overlay), live controls (detection confidence, tiling, hotspot blend, SR
model, colorizer), victim count, per-branch latency, model quality and downloads
(overlay PNG, all stages + JSON report as a zip). Batch tab: many frames at once,
a gallery, a per-frame CSV and one zip with every overlay.
"""
import argparse
import csv
import json
import tempfile
import threading
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from infravision.config import ROOT, load_config, merge
from infravision.data.pop import POP_ROOT
from infravision.io import ThermalImageError, list_images, read_rgb, read_thermal, save_rgb
from infravision.pipeline import Pipeline

POP_TEST = POP_ROOT / "test" / "JPEGImages"
POP_YOLO_TEST = ROOT / "human_detection" / "datasets" / "pop" / "images" / "test"
PAIRED = ROOT / "data" / "paired_disaster"
MAX_EXAMPLES = 200


def pop_examples():
    folder = POP_TEST if POP_TEST.is_dir() else POP_YOLO_TEST
    files = list_images(folder) if folder.is_dir() else []
    step = max(1, len(files) // MAX_EXAMPLES)
    return {f"POP test / {p.name}": p for p in files[::step][:MAX_EXAMPLES]}


def paired_examples():
    """Synthetic disaster pairs: thermal -> (clean RGB, [degraded RGB variants])."""
    manifest = PAIRED / "manifest.csv"
    if not manifest.exists():
        return {}, {}
    pairs = {}
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    # Test pairs when the full build is there; otherwise whatever split exists.
    split = "test" if any(r["split"] == "test" for r in rows) else "val"
    for r in rows:
        if r["split"] != split:
            continue
        t = PAIRED / r["thermal"].replace("\\", "/")
        e = pairs.setdefault(r["id"], dict(thermal=t, rgb=PAIRED / r["rgb"].replace("\\", "/"), degraded=[]))
        e["degraded"].append((f"{r['condition']} ({r['severity']})", PAIRED / r["rgb_degraded"].replace("\\", "/")))
    ids = sorted(i for i, e in pairs.items() if e["thermal"].exists())
    step = max(1, len(ids) // MAX_EXAMPLES)
    names = {f"Synthetic disaster (LLVIP {split}) / {i}": pairs[i]["thermal"] for i in ids[::step][:MAX_EXAMPLES]}
    by_thermal = {str(pairs[i]["thermal"]): pairs[i] for i in ids}
    return names, by_thermal


COLORIZERS = {"ThermoVision U-Net (default)": "thermovision",
              "InfraVision tiny (old, for comparison)": "infravision_tiny"}
# Bicubic first (the default): EDSR works on the 2x frame, ~60 s per full POP frame on a laptop CPU.
SR_MODELS = {"Bicubic x2 (no model, fast)": "bicubic", "EDSR x2 (trained on POP, ~1 min/frame on CPU)": "edsr",
             "Off (1x, quickest on CPU)": "off"}
PALETTES = ["White-hot (people bright)", "Black-hot (people dark)"]
TILES = {"Off (whole frame)": 0, "2 x 2 tiles": 2, "3 x 3 tiles (small people)": 3}


class Pipelines:
    """One pipeline per (colorizer, SR model), built the first time it is picked.

    Detection confidence, tiling and hotspot blend are cheap per-run settings, so
    they are set on the cached pipeline under a lock instead of building a new one.
    """

    def __init__(self, config=None, overrides=None):
        self.config, self.over, self.cache = config, overrides or {}, {}
        self.lock = threading.Lock()

    def get(self, colorizer, sr):
        key = (colorizer, sr)
        if key not in self.cache:
            over = merge(self.over, {"colorizer": {"backend": colorizer}})
            if sr == "off":
                over = merge(over, {"sr": {"scale": 1}})
            else:
                over = merge(over, {"sr": {"backend": sr}})
            self.cache[key] = Pipeline(self.config, over)
        return self.cache[key]

    def run(self, path, colorizer, sr, conf, tiles, hotspot, black_hot=False):
        p = self.get(colorizer, sr)
        # Models were trained on white-hot frames (warm = bright); black-hot footage is flipped first.
        image = 255 - read_thermal(path)[0] if black_hot else path
        with self.lock:
            p.detector.conf, p.detector.tile = float(conf), int(tiles)
            p.enhancer.post_cfg = {**p.cfg.get("color_post", {}), "hotspot_alpha": float(hotspot) / 100}
            return p, p.run(image)


def checkpoint_metrics(path):
    """Validation scores saved in a training checkpoint (None for older files)."""
    if not path:
        return None
    try:
        m = torch.load(path, map_location="cpu", weights_only=False).get("metrics")
    except Exception:
        return None
    return m if isinstance(m, dict) else None


_METRICS = {}   # checkpoint path -> stored validation scores


def quality_lines(p, cache=_METRICS):
    """Model quality from the checkpoints: SR PSNR/SSIM vs bicubic, colorizer PSNR/SSIM/saturation."""
    lines = []
    sr_path = p.enhancer.sr.path
    if sr_path not in cache:
        cache[sr_path] = checkpoint_metrics(sr_path)
    m = cache[sr_path]
    if m and "val_psnr" in m:
        b = m.get("val_psnr_bicubic")
        lines.append(f"SR on POP val: **{m['val_psnr']:.2f} dB** PSNR, SSIM {m.get('val_ssim', 0):.4f}"
                     + (f" (bicubic {b:.2f} dB)" if b else ""))
    else:
        lines.append(f"SR: {p.enhancer.sr.name} (no validation scores stored)")
    col = p.enhancer.colorizer
    col_path = getattr(col, "path", None) or getattr(getattr(col, "adapter", None), "path", None)
    if col_path not in cache:
        cache[col_path] = checkpoint_metrics(col_path)
    m = cache[col_path]
    if m and "val_psnr" in m:
        lines.append(f"Colorizer on LLVIP val: **{m['val_psnr']:.2f} dB** PSNR, SSIM {m.get('val_ssim', 0):.3f}, "
                     f"saturation {m.get('val_saturation', 0):.3f}")
    else:
        lines.append(f"Colorizer: {col.name}")
    return lines


def report(path, p, r, settings):
    d, t = r["detections"], r["timings"]
    return dict(image=str(path), people=d["count"],
                boxes=[[round(float(v), 1) for v in b] for b in d["boxes"]],
                scores=[round(float(s), 3) for s in d["scores"]],
                input_size=list(d["input_size"]), output_size=list(d["output_size"]),
                timings_ms={k: round(v * 1000, 1) for k, v in t.items()},
                models=p.describe(), settings=settings, notes=r["notes"])


def badge(count):
    color = "#c62828" if count else "#455a64"
    word = "PERSON" if count == 1 else "PEOPLE"
    return (f"<div style='background:{color};color:white;border-radius:10px;padding:14px;text-align:center'>"
            f"<div style='font-size:44px;font-weight:700;line-height:1'>{count}</div>"
            f"<div style='font-size:15px;letter-spacing:2px'>{word} DETECTED</div></div>")


def latency_table(t):
    ms = lambda k: f"{t.get(k, 0) * 1000:.0f}"
    return "\n".join([
        "| stage | ms |", "|---|---|",
        f"| A: denoise | {ms('pre_enhance')} |",
        f"| A: super-resolution | {ms('super_resolution')} |",
        f"| A: CLAHE after SR | {ms('post_clahe')} |",
        f"| A: colorize | {ms('colorize')} |",
        f"| A: colour boost + hotspots | {ms('color_post')} |",
        f"| **Branch A** | **{ms('branch_a')}** |",
        f"| **Branch B: detection** (parallel) | **{ms('branch_b')}** |",
        f"| fusion | {ms('fusion')} |",
        f"| **total** | **{ms('total')}** |",
    ])


def build(pipes, default_colorizer="thermovision"):
    pops = pop_examples()
    synth, by_thermal = paired_examples()
    choices = {**pops, **synth}
    pipe = pipes.get(default_colorizer, "bicubic")
    info = pipe.describe()
    det_cfg, post_cfg = pipe.cfg["detection"], pipe.cfg.get("color_post", {})
    default_col = next((k for k, v in COLORIZERS.items() if v == default_colorizer), next(iter(COLORIZERS)))
    default_tiles = next((k for k, v in TILES.items() if v == int(det_cfg.get("tile", 0))), next(iter(TILES)))

    def paired_rgb(path, which):
        e = by_thermal.get(str(path))
        if not e:
            return None, ""
        if which == "clean" or not e["degraded"]:
            p, label = e["rgb"], "real RGB (clean)"
        else:
            label, p = e["degraded"][0]
            label = f"real RGB with synthetic {label}"
        return (read_rgb(p) if p.exists() else None), label

    def settings_of(colorizer, sr, conf, tiles, hotspot, palette):
        return dict(colorizer=COLORIZERS.get(colorizer, default_colorizer), sr=SR_MODELS.get(sr, "bicubic"),
                    conf=float(conf), tiles=TILES.get(tiles, 0), hotspot_percent=float(hotspot),
                    black_hot=palette == PALETTES[1])

    def run_one(path, s):
        try:
            return pipes.run(path, s["colorizer"], s["sr"], s["conf"], s["tiles"], s["hotspot_percent"],
                             s["black_hot"])
        except ThermalImageError as e:
            raise gr.Error(str(e))
        except Exception as e:   # e.g. a checkpoint that does not load
            raise gr.Error(f"{type(e).__name__}: {e}")

    def process(upload, pick, which, colorizer, sr, conf, tiles, hotspot, palette):
        path = upload or (choices.get(pick) if pick else None)
        if not path:
            raise gr.Error("Upload a thermal image or pick one from the list.")
        s = settings_of(colorizer, sr, conf, tiles, hotspot, palette)
        p, r = run_one(path, s)
        rep = report(path, p, r, s)
        out = Path(tempfile.mkdtemp(prefix="infravision_"))
        stem = Path(path).stem
        png = out / f"{stem}_overlay.png"
        save_rgb(png, r["fused"])
        bundle = out / f"{stem}_infravision.zip"
        with zipfile.ZipFile(bundle, "w") as z:
            for name, img in [("1_thermal", r["thermal"]), ("2_super_resolved", r["sr"]),
                              ("3_colorized", r["colorized"]), ("4_overlay", r["fused"])]:
                f = out / f"{stem}_{name}.png"
                save_rgb(f, np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img)
                z.write(f, f.name)
            z.writestr(f"{stem}_report.json", json.dumps(rep, indent=2))
        rgb, rgb_label = paired_rgb(path, which)
        d = r["detections"]
        info_md = "\n".join(
            [f"Input {d['input_size'][0]}x{d['input_size'][1]} → output {d['output_size'][0]}x{d['output_size'][1]}",
             ""] + [f"- {q}" for q in quality_lines(p)] + [f"- Detector: {p.detector.name}"]
            + (["", "**Notes:**"] + [f"- {n}" for n in r["notes"]] if r["notes"] else []))
        return (r["thermal"], r["sr"], r["colorized"], r["fused"], badge(d["count"]), latency_table(r["timings"]),
                info_md, [str(png), str(bundle)],
                gr.update(value=rgb, label=rgb_label or "real RGB (only for synthetic samples)"))

    def process_batch(files, colorizer, sr, conf, tiles, hotspot, palette, progress=gr.Progress()):
        if not files:
            raise gr.Error("Add one or more thermal images.")
        s = settings_of(colorizer, sr, conf, tiles, hotspot, palette)
        out = Path(tempfile.mkdtemp(prefix="infravision_batch_"))
        rows, gallery, reports = [], [], []
        for f in progress.tqdm(files, desc="frames"):
            path = getattr(f, "name", f)
            p, r = run_one(path, s)
            png = out / f"{Path(path).stem}_overlay.png"
            save_rgb(png, r["fused"])
            gallery.append((str(png), f"{Path(path).name}: {r['detections']['count']} people"))
            rep = report(path, p, r, s)
            reports.append(rep)
            rows.append(dict(image=Path(path).name, people=rep["people"],
                             max_score=max(rep["scores"], default=0), total_ms=rep["timings_ms"]["total"]))
        table = out / "summary.csv"
        with open(table, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        bundle = out / "infravision_batch.zip"
        with zipfile.ZipFile(bundle, "w") as z:
            for g, _ in gallery:
                z.write(g, Path(g).name)
            z.write(table, table.name)
            z.writestr("reports.json", json.dumps(reports, indent=2))
        total = sum(r["people"] for r in rows)
        flagged = sum(1 for r in rows if r["people"])
        summary = (f"### {total} people in {flagged} of {len(rows)} frames\n\n"
                   f"Average {sum(r['total_ms'] for r in rows) / len(rows) / 1000:.1f} s per frame")
        return gallery, summary, [str(table), str(bundle)]

    with gr.Blocks(title="InfraVision") as demo:
        gr.Markdown(
            "# InfraVision rescue dashboard\nThermal drone frames → super-resolution → colorization, with people "
            f"detected and warm spots highlighted.\n\nModels: {info['sr']} · {info['colorizer']} · {info['detector']}"
            + ("".join(f"\n\n⚠️ {n}" for n in info["notes"]) if info["notes"] else ""))
        with gr.Row():
            with gr.Column(scale=1):
                colorizer = gr.Dropdown(list(COLORIZERS), value=default_col, label="Colorizer")
                sr = gr.Dropdown(list(SR_MODELS), value=next(iter(SR_MODELS)), label="Super-resolution model")
                conf = gr.Slider(0.05, 0.9, value=float(det_cfg.get("conf", 0.25)), step=0.05,
                                 label="Detection confidence threshold")
                tiles = gr.Radio(list(TILES), value=default_tiles, label="Tiled detection (finds smaller people)")
                hotspot = gr.Slider(0, 100, value=round(100 * float(post_cfg.get("hotspot_alpha", 0.35))), step=5,
                                    label="Thermal hotspot blend (%)")
                palette = gr.Radio(PALETTES, value=PALETTES[0], label="Input palette (black-hot gets flipped)")
            with gr.Column(scale=3):
                with gr.Tab("Single image"):
                    with gr.Row():
                        upload = gr.Image(type="filepath", label="Upload a thermal image", sources=["upload"],
                                          height=200)
                        with gr.Column():
                            pick = gr.Dropdown(list(choices), label="…or pick a POP test image / synthetic sample",
                                               value=next(iter(choices), None))
                            which = gr.Radio(["degraded", "clean"], value="degraded",
                                             label="Real RGB panel (synthetic samples): disaster-degraded or clean")
                            run_btn = gr.Button("Run InfraVision", variant="primary")
                    with gr.Row():
                        s1 = gr.Image(label="1. Raw thermal", interactive=False)
                        s2 = gr.Image(label="2. Super-resolved thermal", interactive=False)
                    with gr.Row():
                        s3 = gr.Image(label="3. AI colorized", interactive=False)
                        s4 = gr.Image(label="4. Tactical overlay (people + hotspots)", interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            count = gr.HTML(badge(0))
                            files = gr.File(label="Downloads: overlay PNG · all stages + report (zip)",
                                            file_count="multiple")
                        latency = gr.Markdown()
                        quality = gr.Markdown()
                    # Always shown: Gradio 6 does not bring back a panel created with visible=False.
                    p_rgb = gr.Image(label="real RGB (only for synthetic samples)", interactive=False, height=240)
                with gr.Tab("Batch"):
                    batch_in = gr.File(label="Thermal frames", file_count="multiple", file_types=["image"])
                    batch_btn = gr.Button("Run on all frames", variant="primary")
                    batch_sum = gr.Markdown()
                    batch_gal = gr.Gallery(label="Overlays", columns=4, height=420)
                    batch_files = gr.File(label="Downloads: summary CSV · all overlays + reports (zip)",
                                          file_count="multiple")
        gr.Markdown("Colours are estimated by a model trained on LLVIP street scenes; they are not temperatures. "
                    "Hotspots mark the warmest pixels of each frame (not only people).")
        controls = [colorizer, sr, conf, tiles, hotspot, palette]
        run_btn.click(process, [upload, pick, which] + controls,
                      [s1, s2, s3, s4, count, latency, quality, files, p_rgb])
        batch_btn.click(process_batch, [batch_in] + controls, [batch_gal, batch_sum, batch_files])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--colorizer", default=None, choices=list(COLORIZERS.values()),
                   help="colorizer selected at start (default: colorizer.backend in the config)")
    p.add_argument("--yolo", default=None, help="YOLO weights to use instead of the config list (e.g. yolo11s.pt)")
    args = p.parse_args()
    over = {"device": args.device} if args.device else {}
    if args.yolo:
        over["detection"] = {"weights": [args.yolo]}
    backend = args.colorizer or load_config(args.config)["colorizer"]["backend"]
    build(Pipelines(args.config, over), backend).launch(server_name=args.host, server_port=args.port,
                                                         share=args.share)


if __name__ == "__main__":
    main()
