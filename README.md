# InfraVision

InfraVision is a PyTorch research project for thermal (infrared) imagery. It
enhances a thermal image and detects the people in it, with an optional
thermal-to-colour translation step.

![Original, SwinIR-enhanced and detection output](integrated_outputs/infravision_comparison.png)

## Pipeline

| Stage | Folder | Model | Role |
|---|---|---|---|
| Enhancement | `SR/` | EDSR and a SwinIR-style refinement network | Restores detail in degraded thermal images |
| Person detection | `human_detection/` | YOLO11s (Ultralytics) | Finds people in thermal images |
| Colorization | `colorization/` | CycleGAN | Translates single-channel thermal into RGB (standalone) |

`integrate_infravision.py` ties the first two together:

1. Loads a thermal test image.
2. Enhances it with SwinIR. The enhanced image is used **for display only**.
3. Runs YOLO on the **original** thermal image. YOLO was trained on raw
   thermal frames, and feeding it the SwinIR output roughly halved recall in
   testing.
4. Draws the detections on the enhanced image and saves the results to
   `integrated_outputs/`.

Colorization is not wired into the integrated script yet; run it on its own
(see below).

## Repository layout

```
InfraVision/
├── integrate_infravision.py   # end-to-end demo: enhance + detect
├── integrated_outputs/        # demo results
├── SR/                        # super-resolution / enhancement
│   ├── models/                # edsr.py, swinir_refine.py
│   ├── prepare_dataset.py     # DIV2K degradation pairs
│   ├── prepare_pop_dataset.py # POP thermal degradation pairs
│   ├── train*.py              # EDSR, SwinIR and EDSR+Swin training variants
│   └── evaluate*.py, test*.py # PSNR / SSIM evaluation
├── human_detection/           # YOLO11s person detection
│   ├── datasets/              # convert_flir.py, convert_pop.py (COCO -> YOLO)
│   ├── organize_dataset.py    # builds thermal_person train/val/test split
│   ├── train.py, detect.py, evaluate.py
│   └── data.yaml
├── colorization/              # CycleGAN thermal -> RGB
│   ├── train_cyclegan.py
│   └── colorize.py
└── colab_training/            # Colab / Kaggle notebooks for long YOLO runs
```

## Installation

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for training;
the demo also runs on CPU.

```bash
git clone https://github.com/elzasabu/InfraVision.git
cd InfraVision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# PyTorch: pick the command for your CUDA version from https://pytorch.org
pip install torch torchvision

pip install ultralytics opencv-python pillow numpy matplotlib tqdm scikit-image
```

The Colab notebooks pin `ultralytics==8.4.177`; use the same version if you
want results that match the trained weights exactly.

> **Note:** the committed `requirements.txt` is UTF-16 encoded, includes many
> packages this project does not use, and is missing `ultralytics` and
> `opencv-python`. Use the commands above until it is cleaned up.

## Model weights and datasets

Weights (`*.pt`, `*.pth`) and datasets are excluded by `.gitignore`, so a
fresh clone cannot run anything until you put them in place. Paths below are
relative to the repository root.

### Weights

| File | Used by | Produced by |
|---|---|---|
| `human_detection/runs/thermal/colab_resumed/weights/best.pt` | `integrate_infravision.py` | `colab_training/train_thermal_yolo_colab.ipynb` (FLIR thermal, training resumed on Colab) |
| `human_detection/runs/pop/baseline/weights/best.pt` | `human_detection/detect.py` | `human_detection/train.py` (POP) |
| `SR/checkpoints_pop/swinir_direct/swinir_direct_epoch_20.pth` | `integrate_infravision.py` | `SR/train_swinir.py` |
| `colorization/checkpoints/cyclegan/colorizer_G_AB.pth` | `colorization/colorize.py` | `colorization/train_cyclegan.py` |

`yolo11s.pt` (the pretrained starting point for `human_detection/train.py`) is
downloaded automatically by Ultralytics.

### Datasets

| Path | Contents | Source |
|---|---|---|
| `POP/{train,test}/JPEGImages/` and `POP/{train,test}/Annotations/{split}.json` | POP drone thermal images with COCO annotations | POP dataset |
| `human_detection/datasets/flir_align/` | `thermal/{train,test}/`, `visible/{train,test}/`, `coco_annotations/` | FLIR ADAS aligned (thermal + visible pairs) |
| `SR/dataset/DIV2K_train_HR/` | DIV2K high-resolution images (optional, for the original SR experiments) | DIV2K |

Derived datasets are generated from these with the scripts below.

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

## Checkpoint folder (new pipeline)

The `infravision` package reads and the training scripts write one folder:

```
checkpoints/
  sr/edsr_best.pth             python -m infravision.train.sr          (+ metrics.json)
  colorizer/colorizer_best.pth python -m infravision.train.colorizer   (+ metrics.json)
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

## Running the integrated demo

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

## New pipeline: demo app and batch CLI (runs on CPU)

```bash
python app.py                      # Gradio app at http://127.0.0.1:7860 (add --share for a public link)
python -m infravision.infer --input POP/test/JPEGImages --out outputs/pop_test --limit 20
```

The app shows thermal input | fused output (colorized + enhanced, people
boxed, people count) | real RGB (only for images that have a paired RGB, such
as the synthetic disaster samples). It lists POP test frames and synthetic
samples to pick from, shows per-branch timings, and offers the fused image as
a download. "Fast" mode skips the 2x upscale. On a 4-core CPU a 1280x1024
POP frame takes about 45 s at 2x and 15 s in fast mode.

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
a disconnected session can just be run again. `SMOKE = True` in the first
cell checks the whole notebook in a few minutes.

## Training and evaluating each stage

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
