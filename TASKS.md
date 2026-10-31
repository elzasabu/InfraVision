# TASKS.md: InfraVision Rev1 task board

**How to use:** start a new chat with "Do task C2 from TASKS.md". Each task stands on its own. When a task is finished, tick it, add a one-line result under it, and commit.
Owner **U** = the user (accounts, downloads, uploads, pushing). Owner **C** = Claude (code, local runs).
GitHub issues: progress log #15 · U1 #16 · C3 #17 · C5 #18 · C4 #19 · C1 #20 · C6 #21 · C7 #22. Comment on or close the matching issue when a task finishes, and push.

## Context (read first)

- **Layout:** `Source/` holds `InfraVision/` (the git repo, branch `feature/rev1`, remote `elzasabu/InfraVision`), `datasets/` (POP, LLVIP, outside git) and `docs/` (report PDF and plan docx, outside git).
- **Python:** `uv`, with a venv in `InfraVision/.venv` (CPU torch). Run tests with `.venv/bin/python -m pytest -q`.
- **Dataset paths:** set in `configs/infravision.yaml` under `datasets:`. The first folder that exists is used.

### Facts that override CLAUDE.md where they conflict
- **Hardware:** the laptop has no NVIDIA GPU. All training runs on Kaggle (notebooks are in `colab_training/`). Inference on CPU is fine.
- **Latency:** up to about 60 s per frame is acceptable. **Quality beats speed** (user decision, 2026-10-31).
- **No trained weights exist anywhere.** The team has none. Every model trains from scratch (YOLO starts from COCO `yolo11s.pt`).
- **Already done, don't rebuild:**
  - `device: auto` device selection
  - FP16 autocast on CUDA
  - N×N tiled detection: `detection.tile` in the config
  - the Gradio app: `app.py`
- **Colorizer:** the ThermoVision exp7 U-Net is merged (`infravision/colorizer.py`) but has **no weights and no training code** in this repo.
- **Priority order:** detection first, then SR, then the colorizer, then the dashboard.

## Models: what each does and where it trains

All training runs on Kaggle (free GPU, about 30 GPU hours a week **per account**, at most 12 h per session). Inference runs on the laptop CPU.

| Model | Job in the pipeline | Trains on | Notebook / task | Before | Target |
|---|---|---|---|---|---|
| YOLO11s | finds people (Branch B) | POP (drone thermal, person boxes) | `colab_training/train_pop_yolo_kaggle.ipynb` (U1) | 0.037 mAP@0.5 (FLIR street model) | > 0.65 mAP@0.5 on POP test |
| EDSR x2 | sharpens the thermal frame (Branch A) | POP crops | `colab_training/train_sr_kaggle.ipynb` (U6) | old checkpoint 40.69 dB, worse than bicubic's 43.02 dB | beat bicubic PSNR/SSIM on POP val |
| ThermoVision U-Net | thermal → colour (Branch A) | LLVIP pairs (later maybe WiSARD / NII-CU) | `colab_training/train_colorizer_kaggle.ipynb` (U5) | false-colour map: 9.06 dB PSNR, 0.21 SSIM | well above the baselines; saturation > 0.18 after C4 |

The targets are goals, not promises. Real numbers go into `docs/RESULTS.md` after each run.

## User tasks

**Every training run now stops itself early if it is broken** (NaN loss, EDSR more than 3 dB below bicubic, colorizer output flat, YOLO mAP50 < 0.05 at epoch 5). A run that fails in the first minutes with `HEALTH CHECK FAILED` means: send Claude the log.


- [ ] **U1: train YOLO on Kaggle**
  1. Kaggle account: Settings → Phone verification.
  2. Datasets → New Dataset: upload `InfraVision/human_detection/pop_yolo_colab.zip` (private).
  3. Import `colab_training/train_pop_yolo_kaggle.ipynb`, Add Input → that dataset, Accelerator → GPU, Run All.
  4. Download `pop_1024.zip` from the Output tab. Copy `weights/best.pt` into `InfraVision/checkpoints/yolo/best.pt`.
  - Done when: `checkpoints/yolo/best.pt` exists. Then hand C6 to Claude.
- [x] **U2: get the full LLVIP dataset**
  - Done 2026-10-31: 12025 train / 3463 test pairs at `datasets/LLVIP`.
  - The current copy has an empty `infrared/` folder and only 9251 of 12025 visible train images.
  - Get it from the official repo, github.com/bupt-ai-cz/LLVIP (download links are in its README).
  - Put it at `datasets/LLVIP/{infrared,visible}/{train,test}`.
  - Done when: `infrared/train` and `visible/train` have the same file count (~12k), and the test folders have ~3.4k each.
- [ ] **U3: (optional, only if C5 colours look wrong on POP) download aerial thermal+RGB pairs**
  - WiSARD: sites.google.com/uw.edu/wisard
  - NII-CU MAPD: www.nii-cu-multispectral.org
  - Check each licence. Put them at `datasets/WiSARD/`, `datasets/NII-CU/`, then tell Claude the folder layout.
- [ ] **U5: train the colorizer on Kaggle** (teammate account is fine)
  - Run 1 (2026-10-31) failed: NaN from epoch 1 (frozen ImageNet BatchNorm blew activations past fp16), output a flat brown fill. Fixed; **re-run without attaching the old output**.
  1. Upload `datasets/llvip_colorizer.zip` as a private Kaggle dataset.
  2. New notebook → File → Import Notebook → `colab_training/train_colorizer_kaggle.ipynb`. Add Input → that dataset. GPU T4, Internet on.
  3. Save Version → Save & Run All (Commit).
  4. Output tab → `colorizer_llvip.zip` → copy `thermovision_best.pth` to `InfraVision/checkpoints/colorizer/thermovision_llvip_v1.pth`.
- [ ] **U6: train EDSR on Kaggle** (third account is fine)
  - Run 1 (2026-10-31) failed: stuck at 5.6 dB for all 30 epochs (outputs went all-white; the [0,1] clamp killed the gradient). Fixed in the training code; **re-run without attaching the old output** (it would resume the broken `last.pth`).
  - Run 2 (2026-10-31, fixed code) failed: all weights NaN from epoch 1 under fp16. Now trains in fp32 (no `--amp`, loss always fp32), ~2–3 h. Re-import the notebook.
  1. Upload `datasets/pop_raw.zip` (673 MB, original POP layout) as a private Kaggle dataset.
  2. New notebook → File → Import Notebook → `colab_training/train_sr_kaggle.ipynb`. Add Input → that dataset. GPU T4, Internet on.
  3. Save Version → Save & Run All (Commit).
  4. Output tab → `edsr_pop.zip` → copy `edsr_best.pth` to `InfraVision/checkpoints/sr/edsr_best.pth`. The last cell says whether it beats bicubic. If it doesn't, set `sr.backend: bicubic`.
- [ ] **U7: disaster datasets** (replaces U3)
  - HIT-UAV (thermal drone, person boxes, CC BY 4.0): `git clone https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset` into `datasets/HIT-UAV/`.
  - WiSARD (wilderness SAR, thermal + RGB, MIT-style licence), from sites.google.com/uw.edu/wisard: first the **971.6 MB Multi Modal Sample** into `datasets/WiSARD/`. Only get the 40.5 GB full set if the sample shows a clear gain.
  - Then hand C8 to Claude.
- [ ] **U8: real flood test frames** (optional, later)
  - 50–100 thermal drone frames of people in water (news / rescue footage) in `datasets/flood_real/`. Claude pre-labels them with YOLO; you only fix the boxes (CVAT or Roboflow).
- [ ] **U4: push when asked:** `git push -u origin feature/rev1` (from `InfraVision/`).

## Claude tasks

- [x] **C0: Phase 0.** FP16 autocast, `scripts/benchmark_baseline.py`, merged the ThermoVision branch, dataset paths in config, POP → YOLO zip for Kaggle.
- [x] **C1: flood-occlusion augmentation** (RUN_ORDER 2.2)
  - Done 2026-10-31: `scripts/prepare_flood_aug.py` adds 1664 `flood_*` train copies (30%) and a 900-frame `test_flood` split + `data_flood.yaml`. Water level is taken from the ground around each person, as a soft blob with a wavy edge. Grid at `results/flood_aug_grid.jpg`. `make_pop_zip.py` includes them (zip now 1.01 GB). Used from the next YOLO run (the running U1 uses the old zip).
  - `scripts/prepare_flood_aug.py`: offline copies of ~30% of POP train frames.
  - In each copy, the lower 30–60% of every person box is filled with cool, water-like intensity plus ripple noise. Labels stay unchanged.
  - Add the copies to `human_detection/datasets/pop`, rebuild the zip, and build a flood-occluded test set from POP test.
  - Done when: the augmented images look right by eye (save a grid) and `make_pop_zip.py` includes them.
- [x] **C2: Branch A reorder** (RUN_ORDER 1.1–1.2)
  - Done 2026-10-31: `denoise_method: bilateral`, `clahe_stage: post` (clip 1.5), `sr.backend: edsr | bicubic`. Pre-enhance went from 472 to 13 ms per POP frame. 47 tests pass. PSNR check waits for C3 weights.
  - New order: `cv2.bilateralFilter` → SR → CLAHE (clip 1.5, 8×8) after SR.
  - Config switch `sr.backend: bicubic | edsr`.
  - Removes the 2.5 s NLMeans step. Tests pass. Update the defaults in `configs/infravision.yaml`.
- [x] **C3: prepare EDSR training on POP** (RUN_ORDER 1.3)
  - Done 2026-10-31: the existing `infravision/train/sr.py` is kept as is (L1 + 0.2·(1−SSIM), Adam 2e-4, cosine, 30 epochs, batch 16; a 2-epoch CPU test already matched bicubic). Added `colab_training/train_sr_kaggle.ipynb` and `datasets/pop_raw.zip`. Smoke-tested from the extracted zip; the checkpoint loads in the pipeline. Training itself = U6.
  - Check `infravision/train/sr.py` against the recipe: 25 epochs, batch 16, AdamW 1e-4, cosine schedule, loss 0.9·L1 + 0.1·(1−SSIM).
  - Make a Kaggle notebook plus a data zip. Smoke-run 1 epoch on CPU.
  - Then **U** runs it on Kaggle. Gate: beats bicubic on POP val (`python -m infravision.evaluate --skip colorizer grid timing`).
- [x] **C4: colorizer post-processing** (RUN_ORDER 3), CPU only
  - Done 2026-10-31: `infravision/colorpost.py`, config `color_post` (chroma 1.8 + L CLAHE 1.5; hotspot top 5% of the SR frame before CLAHE, magma, alpha 0.35). Pipeline returns `colorized` (raw) and `enhanced_rgb` (post-processed). ~0.4–0.6 s per full POP frame on CPU. Grids: `results/color_post_grid.jpg`, `results/color_post_zoom.jpg` (made with a 2-epoch CPU colorizer; people hidden in the colour output become clearly visible). Saturation gate (> 0.18) waits for the U5 weights. 54 tests pass.
  - LAB chroma boost (`chroma_boost: 1.8`) and thermal hotspot fusion (top 5% heat, magma LUT, `alpha: 0.35`).
  - Both are config switches, with tests.
- [x] **C5: colorizer training code** (needs U2)
  - Done 2026-10-31: `infravision/train/thermovision.py` (ImageNet ResNet34 encoder, L1 + edge + VGG + Cb/Cr chroma, logs val saturation), `scripts/pack_llvip.py`, Kaggle notebook. Smoke-tested, with a test. Training itself = U5.
  - Add ThermoVision U-Net training (L1 + edge + VGG + Lab-chroma losses) to `infravision/train/colorizer.py`, plus a Kaggle notebook.
- [ ] **C6: detection evaluation** (needs U1)
  - Sweep imgsz, tiling (`detection.tile`) and TTA on POP val, then pick the best.
  - Report test-set P / R / mAP@0.5 to `results/detection_upgrade.json` and update `docs/RESULTS.md`.
- [ ] **C8: disaster datasets → YOLO + eval** (needs U7)
  - Converters for HIT-UAV and WiSARD into YOLO format (person only, full resolution), plus WiSARD thermal/RGB pairs for the colorizer.
  - Evaluate the POP YOLO on POP test, `test_flood`, HIT-UAV and WiSARD → `results/disaster_eval.json`. Then a combined Kaggle fine-tune.
- [ ] **C7: dashboard** (RUN_ORDER 4)
  - Extend `app.py`: 4-stage view, conf / tiling / hotspot sliders, SR selector, victim count, per-branch ms, export.
  - Check it with the Playwright MCP.
