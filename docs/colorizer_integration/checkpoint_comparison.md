# Colorizer checkpoint comparison (step 1 of the ThermoVision integration)

Date: 2026-10-30, jeas's laptop (Windows 11, CPU only, 12 torch threads). No model was trained for this.
Script: `scripts/compare_checkpoints.py`. Raw numbers: `checkpoint_comparison.json`,
`checkpoint_comparison_per_image.csv`. Grids: `grid_paired.png` (6 pairs), `grid_pop.png` (4 POP frames).

## Evaluation set (fixed)
- **50 LLVIP pairs**: evenly spaced ids of the `data/paired_disaster` **test** split (night street scenes, real RGB).
  None of the ThermoVision checkpoints saw LLVIP. The tiny InfraVision colorizer was trained on 75 pairs of the
  *train* split, so it did not see these either.
- **50 FLIR pairs**: evenly spaced pairs of the ThermoVision **validation** split (`align_validation.txt`), never
  trained on. Overlap caveat: ThermoVision picked its `best.pth` checkpoints with a 100-image subset of this same
  split, so the FLIR numbers slightly favour the `best` checkpoints over `last`. There is no separate FLIR test split.
- **20 POP test frames** (evenly spaced, `POP/test/JPEGImages`): no RGB exists, so only saturation and structure.

Identical preprocessing for every model: grey frame resized to 256 x 256 (`cv2.INTER_AREA`), then the model's own
value range ([0, 1] for ThermoVision, [-1, 1] for InfraVision's ResnetGenerator). Target RGB resized to 256 x 256 the
same way. "tiny (pipeline)" is the same tiny model run the way the old pipeline ran it (CLAHE + denoise, longer side
640); it is shown for reference and is not a separate checkpoint, so it is not ranked.

Checks: the vendored U-Net in `infravision/models/thermovision_colorizer.py` gives **bit-identical** output to
ThermoVision's own `unet_models.load_unet` (max |diff| = 0). Using ThermoVision's PIL bilinear resize instead of
`INTER_AREA` changes exp7 PSNR by +0.05 dB, which is negligible.

Metrics: PSNR (dB), SSIM (Gaussian 11 x 11), LPIPS (AlexNet, torchmetrics; lower is better), saturation = mean HSV
saturation (0 = grey), structure = correlation of Sobel edge magnitude between the thermal input and the output luma.
Time = median CPU seconds per image over 30 frames (15 FLIR + 15 LLVIP) after 3 warm-up frames, with the 256 x 256
resize included and file reading excluded. Peak RAM = peak working set of a fresh process that loads only that
model (about 320 MB of it is Python + torch before the model is loaded).

## Candidates
| Name | File (ThermoVision folder) | Size | Date | Stored training info |
|---|---|---|---|---|
| exp3 | `checkpoints/exp3_chroma_best.pth` | 133 MB | 2026-10-26 | ThermalColorizationNet (transposed decoder), epoch 1, step 600, val L1 (100-img subset) 0.1416 |
| exp4 | `checkpoints/exp4_edge_magnitude_best.pth` | 133 MB | 2026-10-26 | same net, epoch 1, step 600, val L1 0.1411 |
| exp6 | `experiments/exp6_resnet34_unet/best.pth` | 93 MB | 2026-10-26 | ResNet34 U-Net, step 3000, val_total 0.2761 |
| exp6b | `experiments/exp6b_resnet34_random_init/best.pth` | 93 MB | 2026-10-28 | same U-Net, random-init encoder, step 2800, val_total 0.2833 |
| exp7 best | `experiments/exp7_resnet34_unet_continue/best.pth` | 93 MB | 2026-10-29 | exp6 + continuation, step 400 of 1500, val_total 0.2722 (ThermoVision's adopted "best") |
| exp7 last | `experiments/exp7_resnet34_unet_continue/last.pth` | 93 MB | 2026-10-29 | same run, final step 1500 (no val metric stored) |
| tiny | InfraVision `colorizer_best.pth` from the audit (75 pairs x 3 epochs, CPU) | 45 MB | 2026-10-30 | epoch 3, val PSNR 17.16, SSIM 0.373 |

All are generator weights only: no optimizer, EMA or discriminator state in these files.

## Result: 100 paired images (LLVIP + FLIR, mean of the two sets)
| Model | LPIPS rank | SSIM rank | Rank sum | PSNR | SSIM | LPIPS | Saturation | Structure | CPU s/img | Peak RAM MB |
|---|---|---|---|---|---|---|---|---|---|---|
| **exp7 last** | 1 | 2 | **3** | **12.07** | 0.401 | **0.685** | 0.047 | 0.238 | 0.09 | 562 |
| exp6 | 3 | 1 | 4 | 11.89 | **0.402** | 0.693 | 0.046 | 0.246 | 0.09 | 562 |
| exp7 best | 2 | 3 | 5 | 11.60 | 0.395 | 0.690 | 0.047 | 0.257 | 0.09 | 562 |
| exp6b | 5 | 4 | 9 | 10.47 | 0.375 | 0.699 | 0.041 | 0.329 | 0.09 | 561 |
| exp4 | 4 | 6 | 10 | 10.38 | 0.368 | 0.694 | 0.050 | 0.342 | 0.98 | 828 |
| exp3 | 6 | 5 | 11 | 10.28 | 0.371 | 0.718 | 0.052 | 0.299 | 0.98 | 829 |
| tiny | 7 | 7 | 14 | 11.24 | 0.295 | 0.879 | 0.482 | 0.113 | 0.39 | 533 |
| tiny (pipeline) | - | - | - | 11.30 | 0.339 | 0.798 | 0.463 | 0.282 | 2.60 | 835 |

## Per set
| Model | LLVIP PSNR | SSIM | LPIPS | Sat. | Brightness | FLIR PSNR | SSIM | LPIPS | Sat. | POP Sat. | POP Structure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| target (real RGB) | | | | 0.553 | 0.18 | | | | 0.145 | n/a | n/a |
| exp3 | 5.94 | 0.171 | 0.818 | 0.055 | 0.60 | 14.62 | 0.570 | 0.618 | 0.049 | 0.047 | 0.211 |
| exp4 | 6.09 | 0.171 | 0.802 | 0.053 | 0.59 | 14.67 | 0.564 | 0.586 | 0.048 | 0.043 | 0.295 |
| exp6 | 8.49 | 0.210 | 0.821 | 0.053 | 0.47 | 15.29 | 0.593 | 0.566 | 0.038 | 0.039 | 0.173 |
| exp6b | 5.81 | 0.168 | 0.807 | 0.044 | 0.62 | 15.13 | 0.582 | 0.591 | 0.038 | 0.040 | 0.203 |
| exp7 best | 7.63 | 0.195 | 0.806 | 0.053 | 0.51 | **15.56** | **0.596** | 0.573 | 0.041 | 0.037 | 0.168 |
| **exp7 last** | 8.62 | 0.206 | 0.821 | 0.056 | 0.47 | 15.52 | **0.596** | **0.549** | 0.039 | 0.033 | **0.314** |
| tiny | 16.72 | 0.397 | 0.799 | 0.475 | 0.12 | 5.76 | 0.193 | 0.958 | 0.488 | 0.467 | 0.131 |
| tiny (pipeline) | **16.81** | **0.448** | **0.709** | 0.459 | 0.12 | 5.80 | 0.229 | 0.888 | 0.467 | 0.457 | 0.286 |

Paired differences on the 100 images (bootstrap 95 % CI; share of images where exp7 last is better):
- exp7 last minus exp7 best: PSNR +0.48 dB (+0.19..+0.71, 72 %), SSIM +0.0056 (+0.0016..+0.0089, 76 %).
- exp7 last minus exp6: PSNR +0.18 dB (+0.08..+0.29), SSIM -0.0010 (-0.0026..+0.0007): a tie on SSIM.
- exp7 last minus exp3: PSNR +1.79 dB (+1.52..+2.08, 86 %), SSIM +0.030 (+0.027..+0.033, 96 %).
- exp7 last minus tiny: PSNR +0.83 dB (-1.30..+3.03), SSIM +0.106 (+0.043..+0.168).

## Decision
**Rule:** sort by the sum of the LPIPS rank and the SSIM rank on the 100 paired images; ties go to the higher
structure score, then the faster model.
**Chosen: ThermoVision exp7 `last.pth`** (rank sum 3), copied to `checkpoints/colorizer/thermovision_exp7_last.pth`
and used by default. exp7 `best.pth` (ThermoVision's own pick, rank sum 5) is kept selectable as
`checkpoints/colorizer/thermovision_exp7_best.pth` (same architecture, change `colorizer.checkpoint`).

Honest reading:
- **exp3 vs exp7 is not a close call.** exp3 is 1.8 dB PSNR and 0.03 SSIM below exp7 last on the same images (CI far
  from zero), and it is 11x slower (0.98 s vs 0.09 s). exp3 looks "sharper" in the grids because it keeps more
  high-contrast texture, but that texture is not closer to the real RGB. Its different architecture (FSEM + 9
  refinement blocks) would also need much more vendored code, so it is not integrated.
- **exp7 last vs exp6 vs exp7 best are close.** They differ by less than 0.01 SSIM and 0.01 LPIPS; the order among
  them is not robust to a different image set. exp7 last wins the rule and is also the only one whose POP structure
  score holds up (0.31 vs 0.17).
- **All ThermoVision models are washed out.** Saturation is 0.03-0.06 against 0.15 (FLIR) and 0.55 (LLVIP) in the real
  RGB: the output is essentially a grey daylight picture with a faint blue/green tint.
- **Night is painted as day.** On LLVIP the U-Nets output mean brightness 0.47-0.62 against 0.18 in the real night
  images, which is why the tiny model "wins" LLVIP PSNR/SSIM: it outputs one constant dark brown for every frame
  (between-image differences near zero; its saturation of 0.47 is that brown, not real colour). On FLIR the tiny
  model collapses (5.8 dB). Its LPIPS is the worst everywhere.
- **POP is out of domain.** All ThermoVision models were trained on FLIR street scenes. On POP forest frames they
  produce grey-green haze, sometimes with bright sky-blue patches over warm ground (exp3, exp4, exp6b strongest;
  exp7 last weakest, see `grid_pop.png`). The U-Nets also show dark borders on some frames (LLVIP rows in
  `grid_paired.png`, a dark stripe at the right edge on two POP frames).

## Integration checks (step 2.6 and 3)
**What the colorizer sees** (`scripts/order_check.py`, `order_check.json`; 3 LLVIP test + 2 FLIR val frames through
branch A with EDSR x2, scored against the real RGB at 640 x 512):

| colorizer input | detail add-back | PSNR | SSIM | LPIPS | saturation |
|---|---|---|---|---|---|
| **raw frame (default)** | **0.5 (default)** | 11.62 | 0.331 | **0.693** | 0.056 |
| raw frame | 0 | 11.76 | **0.425** | 0.804 | 0.055 |
| CLAHE + denoise + SR frame | 0.5 | 11.56 | 0.325 | 0.703 | 0.054 |
| CLAHE + denoise + SR frame | 0 | **11.80** | 0.421 | 0.799 | 0.053 |

The stage order stays pre-enhance -> SR -> colorizer, but the U-Net reads the **raw** frame (what it was trained on;
slightly better LPIPS and SSIM than the enhanced frame, a tie on PSNR) and paints at the SR size. The 1-channel SR
frame then supplies the size and the added-back detail. Detail 0.5 trades SSIM (-0.09) for LPIPS (-0.11) and visibly
sharper output; it is kept as before and can be set with `colorizer.detail`. Five images are a quick check, not a
benchmark.

**Outputs** (`scripts/verify_outputs.py`, `verify_outputs.json`; CLI on 5 POP + 5 paired frames with each colorizer):
no R/B swap (swapping R and B lowers PSNR on all 5 paired frames), flat black/white pixels <= 0.8 %, no checkerboard
(period-2 power 0.15-2.5 ppm of the image power, the same order as the SR frame it came from), no seams at the SR
tile borders (gradient there within the range of the neighbouring columns). `scripts/box_check.py`: on 3 frames with
15 detections the boxes on the 2x output are exactly the thermal boxes x2 (0 px error); `box_check.png` shows the
crops. `before_after.png`: tiny vs exp7 last on 5 frames (3 paired, 2 POP).
