"""InfraVision demo app (Gradio). Runs on CPU.

    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # temporary public link for a friend
    python app.py --device cpu --config configs/infravision.yaml
    python app.py --colorizer infravision_tiny   # start with the old colorizer (switchable in the UI)

Three panels: thermal input | fused output (colorized + people boxed) | real RGB
(only for images that have a paired RGB frame, e.g. the synthetic disaster set).
"""
import argparse
import csv
import tempfile
from pathlib import Path

import gradio as gr

from infravision.config import ROOT, load_config, merge
from infravision.io import ThermalImageError, list_images, read_rgb, save_rgb
from infravision.pipeline import Pipeline

POP_TEST = ROOT / "POP" / "test" / "JPEGImages"
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


class Pipelines:
    """One pipeline per (colorizer backend, fast mode), built the first time it is picked."""

    def __init__(self, config=None, overrides=None):
        self.config, self.over, self.cache = config, overrides or {}, {}

    def get(self, backend, fast=False):
        key = (backend, fast)
        if key not in self.cache:
            over = merge(self.over, {"colorizer": {"backend": backend}})
            if fast:
                over = merge(over, {"sr": {"scale": 1}})
            self.cache[key] = Pipeline(self.config, over)
        return self.cache[key]


def build(pipes, default_backend="thermovision"):
    pops = pop_examples()
    synth, by_thermal = paired_examples()
    choices = {**pops, **synth}
    pipe = pipes.get(default_backend)
    info = pipe.describe()
    default_label = next((k for k, v in COLORIZERS.items() if v == default_backend), next(iter(COLORIZERS)))

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

    def process(upload, pick, which, speed, colorizer):
        path = upload or (choices.get(pick) if pick else None)
        if not path:
            raise gr.Error("Upload a thermal image or pick one from the list.")
        fast = speed.startswith("Fast") and pipe.enhancer.sr.scale > 1
        try:
            p = pipes.get(COLORIZERS.get(colorizer, default_backend), fast)
            r = p.run(path)
        except ThermalImageError as e:
            raise gr.Error(str(e))
        except Exception as e:   # e.g. a colorizer checkpoint that does not load
            raise gr.Error(f"{type(e).__name__}: {e}")
        rgb, rgb_label = paired_rgb(path, which)
        d, t = r["detections"], r["timings"]
        out = Path(tempfile.mkdtemp(prefix="infravision_")) / f"{Path(path).stem}_fused.png"
        save_rgb(out, r["fused"])
        lines = [
            f"### People detected: **{d['count']}**",
            f"Input {d['input_size'][0]}x{d['input_size'][1]} → output {d['output_size'][0]}x{d['output_size'][1]}",
            "",
            f"Colorizer: {p.enhancer.colorizer.name}",
            "",
            "",
            "| stage | seconds |", "|---|---|",
            f"| Branch A: pre-enhance | {t['pre_enhance']:.2f} |",
            f"| Branch A: super-resolution | {t['super_resolution']:.2f} |",
            f"| Branch A: colorize | {t['colorize']:.2f} |",
            f"| **Branch A total** | **{t['branch_a']:.2f}** |",
            f"| **Branch B: detection (parallel)** | **{t['branch_b']:.2f}** |",
            f"| fusion | {t['fusion']:.2f} |",
            f"| **total** | **{t['total']:.2f}** |",
        ]
        if r["notes"]:
            lines += ["", "**Notes:**"] + [f"- {n}" for n in r["notes"]]
        return (r["thermal"], r["fused"], gr.update(value=rgb, label=rgb_label or "real RGB (none for this image)"),
                "\n".join(lines), str(out))

    with gr.Blocks(title="InfraVision") as demo:
        gr.Markdown(
            "# InfraVision\nThermal infrared super-resolution and colorization with person detection "
            "for disaster response.\n\n"
            f"Models: {info['sr']} · {info['colorizer']} · {info['detector']}"
            + ("".join(f"\n\n⚠️ {n}" for n in info["notes"]) if info["notes"] else ""))
        with gr.Row():
            upload = gr.Image(type="filepath", label="Upload a thermal image", sources=["upload"], height=220)
            with gr.Column():
                pick = gr.Dropdown(list(choices), label="…or pick a POP test image / synthetic sample",
                                   value=next(iter(choices), None))
                which = gr.Radio(["degraded", "clean"], value="degraded",
                                 label="Real RGB panel (synthetic samples): disaster-degraded or clean")
                speed = gr.Radio([f"Full ({pipe.enhancer.sr.scale}x super-resolution)",
                                  "Fast (1x restore only, much quicker on CPU)"],
                                 value=f"Full ({pipe.enhancer.sr.scale}x super-resolution)", label="Quality")
                colorizer = gr.Dropdown(list(COLORIZERS), value=default_label, label="Colorizer")
                run_btn = gr.Button("Run InfraVision", variant="primary")
        with gr.Row():
            p_in = gr.Image(label="Thermal input", interactive=False)
            p_out = gr.Image(label="Fused: colorized + enhanced, people detected", interactive=False)
            # Always shown: Gradio 6 does not bring back a panel created with visible=False.
            p_rgb = gr.Image(label="real RGB (synthetic samples only)", interactive=False)
        with gr.Row():
            stats = gr.Markdown()
            dl = gr.File(label="Download fused image")
        gr.Markdown("Colours are estimated by a model trained on street scenes (FLIR); they are not temperatures "
                    "and can be wrong, e.g. sky-blue patches on POP forest frames.")
        run_btn.click(process, [upload, pick, which, speed, colorizer], [p_in, p_out, p_rgb, stats, dl])
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
    args = p.parse_args()
    over = {"device": args.device} if args.device else {}
    backend = args.colorizer or load_config(args.config)["colorizer"]["backend"]
    build(Pipelines(args.config, over), backend).launch(server_name=args.host, server_port=args.port,
                                                         share=args.share)


if __name__ == "__main__":
    main()
