"""InfraVision demo app (Gradio). Runs on CPU.

    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # temporary public link for a friend
    python app.py --device cpu --config configs/infravision.yaml

Three panels: thermal input | fused output (colorized + people boxed) | real RGB
(only for images that have a paired RGB frame, e.g. the synthetic disaster set).
"""
import argparse
import csv
import tempfile
from pathlib import Path

import gradio as gr

from infravision.config import ROOT
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


def build(pipe, fast_pipe=None):
    pops = pop_examples()
    synth, by_thermal = paired_examples()
    choices = {**pops, **synth}
    info = pipe.describe()

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

    def process(upload, pick, which, speed):
        path = upload or (choices.get(pick) if pick else None)
        if not path:
            raise gr.Error("Upload a thermal image or pick one from the list.")
        try:
            r = (fast_pipe if speed.startswith("Fast") and fast_pipe else pipe).run(path)
        except ThermalImageError as e:
            raise gr.Error(str(e))
        rgb, rgb_label = paired_rgb(path, which)
        d, t = r["detections"], r["timings"]
        out = Path(tempfile.mkdtemp(prefix="infravision_")) / f"{Path(path).stem}_fused.png"
        save_rgb(out, r["fused"])
        lines = [
            f"### People detected: **{d['count']}**",
            f"Input {d['input_size'][0]}x{d['input_size'][1]} → output {d['output_size'][0]}x{d['output_size'][1]}",
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
        return (r["thermal"], r["fused"], gr.update(value=rgb, visible=rgb is not None, label=rgb_label or "real RGB"),
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
                run_btn = gr.Button("Run InfraVision", variant="primary")
        with gr.Row():
            p_in = gr.Image(label="Thermal input", interactive=False)
            p_out = gr.Image(label="Fused: colorized + enhanced, people detected", interactive=False)
            p_rgb = gr.Image(label="real RGB", interactive=False, visible=False)
        with gr.Row():
            stats = gr.Markdown()
            dl = gr.File(label="Download fused image")
        run_btn.click(process, [upload, pick, which, speed], [p_in, p_out, p_rgb, stats, dl])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    over = {"device": args.device} if args.device else {}
    pipe = Pipeline(args.config, over)
    fast = Pipeline(args.config, {**over, "sr": {"scale": 1}}) if pipe.enhancer.sr.scale > 1 else None
    build(pipe, fast).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
