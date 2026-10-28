# Paired thermal / RGB disaster dataset

Builds one aligned thermal + RGB dataset for InfraVision from public paired
datasets, then adds synthetic disaster conditions (smoke, fog, dust, darkness)
to the RGB side. The thermal side is left untouched, because long-wave IR sees
through smoke and darkness. That gap is what the project demonstrates.

Each pair gives you the three panels of the InfraVision view:

| thermal input | clean RGB (training target) | degraded RGB (what a rescuer's camera sees) |
|---|---|---|
| `thermal/<id>.png` | `rgb/<id>.jpg` | `rgb_degraded/<id>__smoke_heavy.jpg` |

Person boxes are written as YOLO labels (class 0) when the source has them.

## 1. Download the sources

None of these can be downloaded automatically (they sit behind forms or Google
Drive / Baidu links), so download them by hand and point the config at them.

| Dataset | What it adds | Pairs | People labels |
|---|---|---|---|
| WiSARD | Drone search and rescue in the wild, visible + thermal | thousands | yes |
| NII-CU MAPD | Drone multispectral person detection, RGB + thermal | ~4k | yes |
| FLAME 2 / FLAME 3 | Drone footage of real wildfires and smoke, RGB + thermal | ~50k frames | no (fire labels) |
| LLVIP | Street-level low-light pedestrians, tightly aligned | ~15k | yes (`llvip` preset) |
| DroneVehicle | Drone RGB + IR, day and night | ~28k | no (vehicles only) |

Pick the ones with people (WiSARD, NII-CU MAPD, LLVIP) for detection and the
three-panel demo. FLAME and DroneVehicle add volume and aerial views for the
colorizer.

There is no public paired dataset of floods, earthquakes or collapsed
buildings. The synthetic conditions cover smoke, dust, fog and darkness, but
not rubble or flood water themselves.

## 2. Write a config

Copy `sources.example.json` to `sources.json` and edit the paths (use forward
slashes on Windows). Source types:

- `llvip`: `root` holds `infrared/`, `visible/` and `Annotations/`.
- `dronevehicle`: `root` holds `train/trainimg(r)`, `val/valimg(r)`, `test/testimg(r)`. The 100 px white border is cropped off.
- `folders`: any dataset with a thermal folder and an RGB folder whose file names
  match. `strip` is a regex removed from both names before matching (for
  example `"_(ir|rgb)$"` pairs `0001_ir.jpg` with `0001_rgb.jpg`). Optional:
  `labels` (YOLO `.txt` or VOC `.xml` per image), `split` (`train`/`val`/`test`),
  `crop` (border in pixels), `max` (cap on pairs). Use one `folders` entry per
  split if the dataset ships its own splits.

The pairs must already be aligned by the dataset (same view, same frame).
The script resizes RGB to the thermal frame but does not register images.

## 3. Build

```bash
pip install numpy pillow
python paired_data/build_paired_dataset.py --config paired_data/sources.json
```

Useful options:

- `--variants 3`: degraded RGB copies per pair (each a different condition/severity).
- `--conditions smoke dust dark smoke+dark`: limit the conditions
  (`fog`, `smoke`, `dust`, `dark`, `smoke+dark`, `dust+dark`).
- `--severities medium heavy`: limit severities (`light`, `medium`, `heavy`).
- `--size 640 512`: resize every pair to one size.
- `--out`: output folder (default `data/paired_disaster/`, which git ignores).

Output:

```
data/paired_disaster/
  {train,val,test}/thermal/       grayscale thermal (.png)
  {train,val,test}/rgb/           clean RGB (.jpg)
  {train,val,test}/rgb_degraded/  synthetic disaster RGB (.jpg)
  {train,val,test}/labels/        YOLO person boxes
  manifest.csv                    id, source, split, paths, condition, severity
  preview.jpg                     thermal | clean RGB | degraded variants
```

Open `preview.jpg` first to check the pairs line up and the effects look right.

## 4. Train the colorizer on it

```bash
cd colorization
python train_cyclegan.py --paired_source disaster
```

This feeds the clean pairs into the existing paired L1 term
(`--paired_weight`). The degraded RGB is not a training target. It is for the
demo panel and for any thermal + RGB fusion or detection experiments
(`DisasterPairedDataset(..., degraded=True)`).

## How the effects work (`degrade.py`)

- **Fog / smoke / dust**: atmospheric scattering, `I = J·t + A·(1 − t)` with
  `t = exp(−density)`. Fog uses a pale airlight and a depth ramp (the top of a
  drone frame is further away). Smoke uses patchy, vertically stretched fractal
  plumes, grey-brown airlight and sometimes a warm fire glow. Dust uses tan
  airlight, desaturation and airborne specks.
- **Darkness**: exposure cut in linear light with photon shot noise, read noise,
  desaturation and a cool shift.
- Combos (`smoke+dark`, `dust+dark`) stack two effects for night-time fires and
  collapses.

Every pair is seeded from its id, so rebuilding gives the same images.
