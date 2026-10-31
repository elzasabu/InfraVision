# InfraVision

**Thermal infrared image super-resolution and colorization with person
detection, for disaster response and search and rescue.**

InfraVision takes one thermal frame (for example from a drone over a flood,
wildfire smoke or a collapsed building) and returns one image: the frame
enhanced, upscaled and colorized, with the people found in it boxed and
counted. POP (drone thermal, people) is the primary dataset; LLVIP and FLIR
supply thermal/RGB pairs for the colorizer. See [docs/RESULTS.md](docs/RESULTS.md)
for current numbers and [DATASETS.md](DATASETS.md) for data sources and
licences.

![10 POP test frames: thermal with ground truth | fused output](docs/results/pop_test_grid.jpg)

## Pipeline

```
thermal frame
 ├─ Branch A: CLAHE + denoise -> EDSR super-resolution -> thermal-to-RGB colorizer
 │            (the ThermoVision colorizer reads the raw frame it was trained on and
 │             paints at the SR size, with the SR detail added back)
 └─ Branch B (in parallel): YOLO11 person detection on the original frame
fusion: boxes rescaled to the upscaled frame, drawn with confidence + "People: N"
```

| Stage | Code | Model | Trained on |
|---|---|---|---|
| Pre-enhance | `infravision/enhance.py` | CLAHE + non-local-means denoise (configurable) | - |
| Super-resolution | `infravision/enhance.py`, `infravision/train/sr.py` | EDSR, x2 by default | POP |
| Colorizer (default) | `infravision/colorizer.py`, `infravision/models/thermovision_colorizer.py` | ThermoVision U-Net (ResNet34 encoder), exp7 checkpoint, 256 x 256 | FLIR ADAS street scenes (not POP) |
| Colorizer (old, selectable) | `infravision/enhance.py`, `infravision/train/colorizer.py` | ResNet generator, L1 + VGG (+ optional SSIM / GAN) | LLVIP pairs (+ synthetic disaster variants), FLIR optional |
| Detection | `infravision/detect.py`, `infravision/train/yolo.py` | YOLO11s (Ultralytics), one class "person" | POP |
| Pipeline + fusion | `infravision/pipeline.py`, `infravision/fusion.py` | runs both branches in a thread pool | - |

Settings live in `configs/infravision.yaml`.

## Repository layout

```
InfraVision/
├── app.py                     # Gradio demo (3 panels), runs on CPU
├── configs/infravision.yaml   # pipeline settings and weight search order
├── infravision/               # the package: pipeline, models, data loaders, CLI
│   ├── pipeline.py, enhance.py, detect.py, fusion.py, infer.py, evaluate.py
│   ├── data/                  # POP loader (flight splits), paired thermal/RGB loader
│   └── train/                 # sr.py, colorizer.py, yolo.py, smoke.py
├── notebooks/train_colab.ipynb  # full training on a free Colab GPU
├── paired_data/               # builds the LLVIP thermal/RGB "disaster" pairs
├── tests/                     # pytest
├── docs/RESULTS.md            # metrics, qualitative grid, CPU timings
├── checkpoints/               # trained weights (git-ignored, see below)
├── SR/, human_detection/, colorization/, colab_training/   # earlier per-stage scripts
└── integrate_infravision.py   # earlier enhance + detect demo
```

## Installation

Python 3.10+ (tested on 3.13). No GPU needed to run the demo or the tests.

```bash
git clone https://github.com/elzasabu/InfraVision.git
cd InfraVision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # 46 tests, about 15 s, no data needed
```

`requirements.txt` pins the versions everything was tested with. For a CUDA
build of PyTorch, install `torch`/`torchvision` from https://pytorch.org
first (same versions), then the requirements.

**CPU-only Linux machine:** plain `pip install` on Linux pulls the large CUDA
build of PyTorch. To get the small CPU build, install torch first:

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(Windows and macOS get the CPU build from `pip install -r requirements.txt` directly.)

**Running the demo needs weights.** A fresh clone has none. Unzip
`infravision_checkpoints.zip` (see [Model weights](#model-weights)) in the repo
folder first; `checkpoints/yolo/best.pt` alone is enough for the app to start
(without the SR and colorizer weights it uses bicubic upscaling and a
false-colour map, and says so on screen).

## Model weights

Weights are **not in git** (`*.pt`, `*.pth` and `checkpoints/` are ignored)
and not in Git LFS. Share them as a GitHub release asset or a zip:

1. Train with `notebooks/train_colab.ipynb`; at the end it saves
   `infravision_checkpoints.zip` to Drive.
2. Upload that zip to a GitHub release (Releases -> Draft a new release ->
   attach the file), or share it from Drive.
3. On another machine, unzip it in the repo root. It creates:

```
checkpoints/sr/edsr_best.pth               (+ metrics.json)
checkpoints/colorizer/colorizer_best.pth   (+ metrics.json)
checkpoints/yolo/best.pt                   (+ thresholds.json, metrics.json)
```

Without them the app still runs: it falls back to older weights if present
(`SR/checkpoints*/`, `human_detection/runs/...`), shows a false-colour map
instead of the colorizer, and says so on screen. YOLO needs at least one
weights file; `yolo11s.pt` is downloaded by Ultralytics when training starts.

**Colorizer weights (ThermoVision U-Net).** These come from the separate
ThermoVision project, not from the Colab notebook. Copy them in by hand (93 MB
each, git-ignored, never commit them):

```
copy <ThermoVision>\experiments\exp7_resnet34_unet_continue\last.pth checkpoints\colorizer\thermovision_exp7_last.pth   (default)
copy <ThermoVision>\experiments\exp7_resnet34_unet_continue\best.pth checkpoints\colorizer\thermovision_exp7_best.pth   (optional)
```

SHA-256: `last.pth` DF322BB7...FF18914C, `best.pth` 70D77BDE...F0775DB4. If the
file is missing, the pipeline prints the path and this copy step and shows a
false-colour map instead; a file that exists but does not match the model
stops with the list of missing/unexpected keys. Put them in the release zip
next to the other weights to share them.

## Colorizer: ThermoVision U-Net

The default colorizer is the ThermoVision exp7 U-Net (ResNet34 encoder, 24.4 M
parameters), chosen from 7 candidate checkpoints by LPIPS + SSIM rank on 100
paired LLVIP/FLIR test images: see
[docs/colorizer_integration/checkpoint_comparison.md](docs/colorizer_integration/checkpoint_comparison.md).
It takes the whole raw thermal frame resized to 256 x 256 in [0, 1] and returns
RGB in [0, 1]; `infravision/colorizer.py` (`ColorizerAdapter.colorize`) handles
8/16-bit and 1/3-channel input, resizing back to any size, and batches.

Switching colorizers:

```bash
python -m infravision.infer --input POP/test/JPEGImages --out outputs/old --colorizer infravision_tiny
python -m infravision.infer --input ... --out ... --colorizer-checkpoint checkpoints/colorizer/thermovision_exp7_best.pth
python app.py --colorizer infravision_tiny     # or use the "Colorizer" dropdown in the app
```

or set `colorizer.backend` (`thermovision` | `infravision_tiny`),
`colorizer.checkpoint`, `colorizer.device` (`auto` | `cpu` | `cuda`, falls back
to CPU with a warning) and `colorizer.input_size` in `configs/infravision.yaml`.

CPU speed (12 threads): the colorizer itself takes about 0.09 s per frame at
256 x 256 and 0.5-0.6 s inside the pipeline at a 2560 x 2048 output (resize and
detail add-back included), against about 2.9 s for the old tiny colorizer. A
whole 1280 x 1024 POP frame takes about 42 s at 2x SR (44 s before) and about
10 s in fast mode; the EDSR stage is almost all of it.

Known limitations (measured, see the comparison file):
- **Washed-out colour.** Saturation is 0.03-0.06 against 0.15-0.55 in real
  photos: the output is a grey "daylight" picture with a faint tint, and the
  SR detail added back (`colorizer.detail`) makes it look partly like an
  embossed thermal image.
- **Trained on street scenes only (FLIR ADAS).** It has never seen POP drone
  views. On POP forest frames it gives grey-green haze and sometimes
  sky-blue patches over warm ground. Night scenes (LLVIP) are painted as day.
  Some frames get dark borders or a dark stripe at an edge.
- **Colours are not temperatures** and are not real-world colours: they are
  the model's guess of what a daylight camera might see, and details can be
  invented. People that stand out in the thermal frame can fade in the
  colour frame. Use the thermal view and the detector, not the colours, for
  decisions.

## Datasets

Nothing is redistributed; download each from its owner (see
[DATASETS.md](DATASETS.md) for links, citations and licences). Paths are
relative to the repo root.

| Path | Contents | Needed for |
|---|---|---|
| `POP/{train,val,test}/JPEGImages/`, `POP/{split}/Annotations/{split}.json` | POP drone thermal frames, COCO person boxes | demo pick list, SR + YOLO training, all metrics |
| `data/paired_disaster/` | LLVIP thermal + RGB pairs with synthetic smoke/fog/dust/dark RGB copies, built by `paired_data/build_paired_dataset.py` | colorizer training, 3-panel demo samples |
| `human_detection/datasets/flir_align/` | FLIR ADAS aligned thermal + visible (optional) | extra colorizer pairs, earlier FLIR scripts |
| `SR/dataset/DIV2K_train_HR/` | DIV2K (optional) | only the earlier SR experiments |

## POP data and splits

POP is the primary dataset: drone thermal frames (1280x1024, stored as three
identical channels) with COCO person boxes. It has **no RGB**, so:

- super-resolution and person detection train and are scored on POP;
- the colorizer trains on aligned thermal/RGB pairs from LLVIP (and optionally
  FLIR) and is only judged by eye on POP.

Frames from one flight are near-duplicates, so splits are by flight, never by
random frame. POP's own splits already do this and are kept as they are:
train F1-F12 (5,548 frames), val F13-F16 (2,320), test T1-T3 (900).

```bash
python -m infravision.data.pop          # counts, box sizes, and a check that no flight is in two splits
python -m infravision.data.pop --resplit --seed 0   # redraw train/val by flight (test is never touched)
```

In code: `from infravision.data.pop import pop_splits, POPDetection, POPSRCrops`.

## Checkpoint folder

The `infravision` package reads and the training scripts write one folder:

```
checkpoints/
  sr/edsr_best.pth             python -m infravision.train.sr          (+ metrics.json)
  colorizer/colorizer_best.pth python -m infravision.train.colorizer   (+ metrics.json)
  colorizer/thermovision_exp7_last.pth   copied from ThermoVision (default colorizer, see above)
  yolo/best.pt                 python -m infravision.train.yolo train  (+ thresholds.json)
  smoke/...                    --smoke runs; never read by the demo
```

Each training script keeps the checkpoint with the best validation score
(PSNR for SR and the colorizer) and writes `metrics.json` next to it.
`configs/infravision.yaml` lists, per model, `checkpoints/...` first and then
the older locations (`SR/checkpoints_pop/`, `SR/checkpoints/`,
`human_detection/runs/...`, `colorization/checkpoints/`), so existing weights
keep working. The folder is git-ignored; weights are shared as a release or
a zip, not committed.

Branch A (enhancement) is configured in `configs/infravision.yaml`:
`pre_enhance` (CLAHE + mild denoise, toggleable), `sr.scale`, and the
colorizer. All training scripts take `--device auto|cpu|cuda` (auto falls back
to CPU) and `--amp` for mixed precision on CUDA.

## Preparing data

```bash
# FLIR: COCO -> YOLO labels, then a train/val/test split for the demo
python human_detection/datasets/convert_flir.py      # -> flir_align/labels/train/
python human_detection/datasets/convert_flir_test.py # -> flir_align/labels/test/
python human_detection/organize_dataset.py          # -> datasets/thermal_person/

# POP: COCO -> YOLO dataset for detection
python human_detection/datasets/convert_pop.py      # -> datasets/pop/

# POP: degraded/clean pairs for super-resolution (run from SR/)
cd SR && python prepare_pop_dataset.py && cd ..     # -> SR/dataset_pop/
```

The SR and colorization scripts use paths relative to their own folder, so run
them from inside `SR/` or `colorization/`.

## Earlier demo (integrate_infravision.py)

With the two weights files and `human_detection/datasets/thermal_person/images/test/`
in place:

```bash
python integrate_infravision.py
```

It uses the first image (sorted by name) in the test folder by default. Pass an
index or an image path to pick another one:

```bash
python integrate_infravision.py 200              # 201st test image
python integrate_infravision.py path/to/image.png
```

Outputs are written to `integrated_outputs/`:

- `enhanced_thermal.png`: SwinIR output
- `infravision_final.png`: enhanced image with person boxes and count
- `infravision_comparison.png`: original, enhanced and detection side by side

Detection uses `imgsz=640`, `conf=0.25` and `iou=0.5`.

## Running the demo app and batch CLI (CPU is fine)

```bash
python app.py                      # Gradio app at http://127.0.0.1:7860 (add --share for a public link)
python -m infravision.infer --input POP/test/JPEGImages --out outputs/pop_test --limit 20
```

The app shows thermal input | fused output (colorized + enhanced, people
boxed, people count) | real RGB (only for images that have a paired RGB, such
as the synthetic disaster samples). It lists POP test frames and synthetic
samples to pick from, shows per-branch timings, and offers the fused image as
a download. "Fast" mode skips the 2x upscale, and the "Colorizer" dropdown
switches between the ThermoVision U-Net and the old tiny colorizer. On a
4-core CPU a 1280x1024 POP frame took about 46 s at 2x and 12 s in fast mode
(see [docs/RESULTS.md](docs/RESULTS.md)).

The CLI writes `<name>_fused.png` per image (`--all` adds the enhanced and SR
frames), `detections.json` and `summary.csv`, and skips unreadable files with
a message.

## Training on Colab (and a CPU smoke test first)

```bash
python -m infravision.train.smoke        # all three trainers, 2 tiny epochs each, on CPU (~1 min)
```

It writes to `checkpoints/smoke/` (never read by the demo) and prints OK or
the error per model. Then train for real on a free GPU with
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb): it clones the
repo, mounts Drive, unzips `POP.zip` and `paired_disaster_colab.zip`
(made by `python paired_data/pack_for_colab.py`, about 4 GB), trains
SR -> colorizer -> YOLO (then tunes the YOLO thresholds and evaluates),
saves each model to Drive as soon as it is done, and zips the
`checkpoints/` folder for the laptop. Stages already on Drive are skipped, so
a disconnected session can just be run again: each model saves its progress
to Drive after every epoch and continues from there (`--resume`). `SMOKE = True`
in the first cell checks the whole notebook in a few minutes.

Step-by-step guide (zips, Drive folder, epochs, what to watch in the logs,
when to stop, what to download): [docs/TRAINING.md](docs/TRAINING.md).

## Earlier per-stage scripts

**Super-resolution** (from `SR/`):

```bash
python train_swinir.py        # SwinIR refinement on dataset_pop
python train_edsr_swin.py     # EDSR + Swin variant
python evaluate_swinir_direct.py
```

**Person detection** (from the repo root):

```bash
python human_detection/train.py      # YOLO11s on POP, 50 epochs, imgsz 640
python human_detection/detect.py     # run the POP model on test images
python human_detection/evaluate.py
```

For longer runs, use the notebooks in `colab_training/` (Colab or Kaggle GPU).
They expect the zipped dataset in Google Drive or as a Kaggle input.

**Colorization** (from `colorization/`):

```bash
python train_cyclegan.py --epochs 100 --batch 2 --size 256
python colorize.py path/to/thermal.jpg --out colorized.png
```

Training uses unpaired POP thermal images and FLIR visible images, plus an
optional supervised loss on aligned FLIR pairs (`--paired_weight`).
