# Training on Colab: step by step

Laptop → Google Drive → Colab (free T4 GPU) → Drive → laptop.

## 1. On the laptop: make the two zips (one time)

Run these in the repo folder, with the venv active.

```bash
git checkout main && git pull        # gets paired_data/pack_for_colab.py and the notebook
python paired_data/pack_for_colab.py
```

| Zip | How to make it | What is inside | Size |
|---|---|---|---|
| `POP.zip` | In Explorer, right-click the `POP` folder → *Send to → Compressed (zipped) folder* | `POP/train`, `POP/val`, `POP/test`, each with `JPEGImages/` and `Annotations/` | ~0.7 GB |
| `paired_disaster_colab.zip` | `python paired_data/pack_for_colab.py` writes it to `data\paired_disaster_colab.zip` | `paired_disaster/manifest.csv` plus the thermal and RGB frames, resized to 640 px | ~4 GB |

`pack_for_colab.py` reads `data/paired_disaster/` (the finished LLVIP build) and
fixes the Windows `\` paths in the manifest for Linux. It takes a while for
15,488 pairs; it prints progress every 2,000 rows.

## 2. Upload to Google Drive

Make a folder called **`InfraVision`** directly in *My Drive* and put both zips in it:

```
My Drive/InfraVision/POP.zip
My Drive/InfraVision/paired_disaster_colab.zip
My Drive/InfraVision/thermal_best.pt      (optional: the FLIR YOLO model, as a starting point)
```

The notebook looks exactly there (`DRIVE_DIR = "/content/drive/MyDrive/InfraVision"`).
It unzips `POP.zip` to `POP/` and `paired_disaster_colab.zip` to
`data/paired_disaster/` inside its own copy of the repo, which are the same
paths the code uses on the laptop.

## 3. Run the notebook

1. Open `notebooks/train_colab.ipynb` in Colab (File → Open notebook → GitHub →
   `elzasabu/InfraVision`).
2. Runtime → Change runtime type → **T4 GPU**.
3. First set `SMOKE = True` in the first code cell and *Run all*. It finishes in a
   few minutes and proves the paths and zips are right.
4. Set `SMOKE = False` and *Run all* again for the real training.

## 4. How long, and how many epochs

The defaults are in the first code cell: `SR_EPOCHS, COLOR_EPOCHS, YOLO_EPOCHS = 20, 15, 40`.
These are **estimates** for a T4, not measured: SR about 1 h, colorizer about 1.5 h, YOLO
about 2.5–3 h. After the first epoch of each model the log prints
`epoch 1 took X min; about Y min left`, so you can check the real time. If it is much
slower, lower that model's epochs. The best epoch is always kept, so more epochs never make
the saved model worse.

## 5. If Colab disconnects

Nothing is lost. Every model saves its progress to Drive after each epoch:

```
My Drive/InfraVision/checkpoints/sr/last.pth          (SR)
My Drive/InfraVision/checkpoints/colorizer/last.pth   (colorizer)
My Drive/InfraVision/yolo_runs/pop_1024/weights/last.pt   (YOLO, plus last_good.pt as a spare copy)
```

Reconnect, choose the T4 again and *Run all*. Each model continues from its last finished
epoch. A finished model has a `DONE` file in its folder and is skipped. To retrain one model
from scratch, put its name in `FORCE`, e.g. `FORCE = ["sr"]`.

## 6. What to watch in the logs

**Super-resolution.** At the start it prints `val PSNR of bicubic input: ~43 dB`. That is the
bar to beat. Each epoch prints `loss`, `val PSNR` and `SSIM`.
- Good: loss goes down, val PSNR climbs above the bicubic number in the first few epochs.
- Bad: val PSNR stays below bicubic after ~5 epochs. Stop and tell me.

**Colorizer.** Each epoch prints `loss`, `val PSNR` and `SSIM`, and the notebook shows
`val_samples_epochNNN.png` (rows: thermal, prediction, real RGB).
- The baselines are about 9 dB PSNR. Good: PSNR well above that, and the middle row in the
  sample image starts to look like the real RGB (sky, road and people in sensible colours).
- Bad: loss flat from the start, or samples all one colour / grey after several epochs.

**YOLO.** Ultralytics prints a table per epoch: `box_loss`, `cls_loss`, `dfl_loss` (should go
down) and on val `P`, `R`, `mAP50`, `mAP50-95` (should go up).
- The FLIR model scores mAP50 0.037 on POP val. Anything far above that is real progress.
- It stops by itself if mAP has not improved for 15 epochs (`patience`).

## 7. When to stop early

- SR or colorizer: if val PSNR has not improved (no `*best*` mark) for about 5 epochs, you can
  stop. Click the stop button, set that model's epochs to the number already done, and *Run
  all*. It sees the training is complete, writes `DONE`, and moves on.
- YOLO: let it run; it stops itself.

## 8. After training: what to download and where it goes

At the end the notebook writes **`My Drive/InfraVision/infravision_checkpoints.zip`**.
Download it and unzip it in the repo folder on the laptop. It creates:

```
checkpoints/sr/edsr_best.pth               + metrics.json
checkpoints/colorizer/colorizer_best.pth   + metrics.json
checkpoints/yolo/best.pt                   + thresholds.json, eval_val.json, eval_test.json
```

The app and CLI pick these up automatically. Then run

```bash
python -m infravision.evaluate --out docs/results
python -m infravision.train.yolo eval --split val --json docs/results/yolo_val.json
```

to refresh the numbers in `docs/RESULTS.md`.
