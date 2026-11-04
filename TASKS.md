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

## Where we are (2026-11-01, 11:00)

- **Next step = U9** (train detector v2 on Kaggle with `human_detection/sar_yolo.zip`). Everything for it is built and dry-run tested.
- **WiSARD full** downloaded (browser; Drive's anonymous quota blocks scripted downloads) → `datasets/WiSARD/WiSARDv1_1.zip` (43.5 GB). Thermal folders extracted to `datasets/WiSARD/WiSARDv1/` (41 IR sequences, 35 labelled, 24k frames, 27.5k person boxes, all 640x512 grey; top-down **and oblique**). No water scenes.
- **Baselines of the current detector** (`checkpoints/yolo/best.pt`): AIResQ (all 1988 frames) mAP50 **0.089 @640**; empty flood blobs detected **26.8%** (`results/flood_decoys_baseline.json`).

### Earlier (2026-11-01, night)

- **Detection** (YOLO, `checkpoints/yolo/best.pt`): POP test mAP50 0.630. On new footage it drops: WiSARD IR sample 0.435. Real test (other machine): cliff/woods frame good (9 boxes on ~7 people, some double counts); **people in water, side angle: 0 found**.
- **Biggest gap = water / flood people.** No public thermal aerial dataset of people in water exists (searched). Plan: (1) dataset request email to the IWT-YOLO authors (Sensors 2024, 24(11) 3321, hejicker@163.com; draft was written, user sends from uni mail + CC supervisor); (2) user collects thermal water-rescue clips (U8), Claude extracts frames + pre-labels (C9); (3) C1 v2 + real-water copy-paste augmentation.
- **Colorizer v1** installed: works, but dark "night photo" look (LLVIP is night-only, memorised ~18 street cameras). Fix later with daytime aerial pairs (WiSARD). Cosmetic, lowest priority.
- **EDSR v3 installed** (`checkpoints/sr/edsr_best.pth`, also in the external-drive weights zip): POP val 46.02 dB vs bicubic 43.02. On 30 full POP test frames it wins 30/30 on degraded (blur/noise/JPEG) input but **loses on a clean x2 downscale (44.40 vs 47.71 dB)**: it over-smooths. Details in `docs/RESULTS.md`. EDSR costs **61 s per full POP frame on CPU**, so the GUI default stays bicubic; EDSR stays selectable.
- **GUI** (`app.py`): bicubic default, new black-hot/white-hot palette switch. Weights zip for other machines: `/run/media/advaithmanoj/SharedData/Projects/Infravision/Data/infravision_weights.zip` (unzip inside the cloned `InfraVision/`). Holds YOLO (POP, sar_v2, sar_v3, COCO near-person), colorizers v1 + v2 and EDSR v3 (updated 2026-11-02). Unzip it in the repo root (`InfraVision/`): the paths inside already start with `checkpoints/`.
- **Downloads:** slow servers (Zenodo ~0.36 MB/s per connection) → `scripts/pdl.sh URL OUT 16` (parallel ranges, chunk-verified, resumable). Google Drive: popular files hit "Quota exceeded" for anonymous/scripted downloads; download in the signed-in browser instead.

- **Large close people (found 2026-11-01):** a low oblique frame with 4 clear people → the POP model found 0 at any imgsz; stock COCO yolo11s found 4/4 (0.87–0.91). The fine-tune forgot large people. Fix now: `detection.near_weights` (COCO model, boxes ≥ 8% of frame height only; no change on POP / HIT-UAV / RGBT in an A/B on 100–150 frames each). Fix in training: LLVIP persons in v3. `checkpoints/yolo/coco_yolo11s.pt` is in the external-drive weights zip (added 2026-11-01).
- **Colorizer (2026-11-01):** retraining on LLVIP won't help (night street scenes). RGBTDronePerson colour/thermal pairs are **not aligned** (different fields of view; checked with edge overlays), so they can't train it either. Use the `palette` / `luma` views; a better colorizer needs registered daytime aerial pairs.

- **Colorizer v3 done (2026-11-04):** `checkpoints/colorizer/thermovision_v3.pth` = dashboard default. Real-camera rotation disagreement about halved on POP / WiSARD / HIT-UAV / AIResQ; blue/green bands gone on 3 of 4 sets (POP band 0.54 → 2.44). CART upright 18.13 → 17.06 dB, upside down 13.26 → 15.77. Generated thermal is blurry, so flood colours are unproven on real thermal; cold meadows can turn blue. Run took ~8 h, not the 4–5 h estimated. Details in `docs/RESULTS.md`.
- **Colorizer v3 ready (2026-11-03), replaces v2.1:** `colab_training/train_colorizer_v3_kaggle.ipynb` in one session: CART + RescueNet (Hurricane Michael, CC0, figshare, MD5-checked) downloaded on Kaggle → reverse model colour → thermal (`thermovision --reverse`, 15 epochs) on the real pairs → pseudo thermal for FloodNet (Hurricane Harvey, 898 photos x 3 views, `datasets/flood_rgb.zip` 123 MB) and RescueNet (898 photos x 3) → colorizer on 5 sets (CART 4, FLAME 2, M3FD 1, FloodNet 2, RescueNet 2) with `--geo_aug 0.5 --thermal_aug`, PatchGAN 0.02. Inputs: `colorizer-extra` + new `flood-rgb`. ~4–5 h. Full dry run on CPU passed.
- **Colorizer v2.1 ready (2026-11-02):** v2 has a **position bias**: it learned "blue = top" from CART's oblique horizon views (held-out frames upside down: 17.9 → 13.4 dB, blue stays in the top third), so nadir drone frames get blue/green bands. `train_colorizer_v21_kaggle.ipynb`: same data, random 90° rotation / flip of half the pairs (`--geo_aug 0.5`), thermal contrast / gamma / blur / noise (`--thermal_aug`), PatchGAN 0.02. ~3 h. Test images: `datasets/colorizer_test_images/` (5 held-out CART thermal frames + real colour + expected v2 sheet).
- **Colorizer v2 done (2026-11-02):** `checkpoints/colorizer/thermovision_v2.pth` = dashboard default. Held-out CART flights: **17.7 dB / SSIM 0.70 vs v1 10.1 / 0.45**, daylight brightness (85 vs real 86, v1 24). Converged (best epoch 31 of 40). Soft textures; blotchier on other thermal cameras. External-drive weights zip updated (21 files: + sar_v3, colorizer v2). Optional v2.1: thermal-input augmentation (contrast / gamma / noise) for other cameras.
- **Colorizer v2 (2026-11-01, top priority):** goal = a *normal colour photo* of the scene from the thermal frame (not a heat palette). Aligned daytime pairs: **CART** (2282, drone, rivers/lakes/coast/desert/forest; downloaded inside the Kaggle notebook), **FLAME 2** clear pre-burn frames (1805 of 53k; smoky fire frames dropped: thermal sees through smoke, colour does not), **M3FD** daytime (3828 of 4200, ground-level streets incl. wet roads, people). `datasets/colorizer_extra.zip` (329 MB) = M3FD + FLAME packs. Notebook `colab_training/train_colorizer_v2_kaggle.ipynb` (PatchGAN, 320 px). Rejected: VTUAV (thermal view ≠ colour view), KAIST preview (night), RGBTDronePerson (not aligned). FLAME 2 zip is ZIP64: open it with Python zipfile or 7z, not `unzip`.

## User tasks

**Every training run now stops itself early if it is broken** (NaN loss, EDSR more than 3 dB below bicubic, colorizer output flat, YOLO mAP50 < 0.05 at epoch 5). A run that fails in the first minutes with `HEALTH CHECK FAILED` means: send Claude the log.


- [x] **U1: train YOLO on Kaggle**
  1. Kaggle account: Settings → Phone verification.
  2. Datasets → New Dataset: upload `InfraVision/human_detection/pop_yolo_colab.zip` (private).
  3. Import `colab_training/train_pop_yolo_kaggle.ipynb`, Add Input → that dataset, Accelerator → GPU, Run All.
  4. Download `pop_1024.zip` from the Output tab. Copy `weights/best.pt` into `InfraVision/checkpoints/yolo/best.pt`.
  - Done when: `checkpoints/yolo/best.pt` exists. Then hand C6 to Claude.
  - Done 2026-10-31 (run with flood copies): val mAP50 0.862 (epoch 31, early stop 47). **POP test: P 0.768, R 0.572, mAP50 0.630** (target 0.65). Flood test 0.990 is invalid: 24% of empty water blobs are detected as people (shortcut), see C1 v2.
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
- [x] **U5: train the colorizer on Kaggle** (teammate account is fine)
  - Done 2026-10-31 (run 2): installed as `checkpoints/colorizer/thermovision_llvip_v1.pth` (best epoch 4: LLVIP val 16.64 dB / SSIM 0.43, baseline false-colour 9.06 / 0.21). POP saturation 0.270 raw → **0.329 after C4** (gate > 0.18 passed), 1.9 s/frame CPU.
  - Limit: it memorised LLVIP's ~18 fixed street cameras (seen scenes 20.5 dB, new scenes 16.7 dB, blurry). LLVIP colour is night photos, so POP output is a dark night-forest look. Rerunning on LLVIP won't fix it; better colour needs daytime aerial pairs with many viewpoints (full WiSARD, registered).
  - Run 1 (2026-10-31) failed: NaN from epoch 1 (frozen ImageNet BatchNorm blew activations past fp16), output a flat brown fill. Fixed; **re-run without attaching the old output**.
  1. Upload `datasets/llvip_colorizer.zip` as a private Kaggle dataset.
  2. New notebook → File → Import Notebook → `colab_training/train_colorizer_kaggle.ipynb`. Add Input → that dataset. GPU T4, Internet on.
  3. Save Version → Save & Run All (Commit).
  4. Output tab → `colorizer_llvip.zip` → copy `thermovision_best.pth` to `InfraVision/checkpoints/colorizer/thermovision_llvip_v1.pth`.
- [x] **U6: train EDSR on Kaggle** (third account is fine)
  - Done 2026-11-01 (run 3, fp32, 196 min): best epoch 29, POP val 46.34 dB / SSIM 0.9868 (bicubic 42.76). Installed. Optional later run 4: mix in undegraded batches so it also beats bicubic on clean input (see RESULTS.md).
  - Run 1 (2026-10-31) failed: stuck at 5.6 dB for all 30 epochs (outputs went all-white; the [0,1] clamp killed the gradient). Fixed in the training code; **re-run without attaching the old output** (it would resume the broken `last.pth`).
  - Run 2 (2026-10-31, fixed code) failed: all weights NaN from epoch 1 under fp16. Now trains in fp32 (no `--amp`, loss always fp32), ~2–3 h. Re-import the notebook.
  1. Upload `datasets/pop_raw.zip` (673 MB, original POP layout) as a private Kaggle dataset.
  2. New notebook → File → Import Notebook → `colab_training/train_sr_kaggle.ipynb`. Add Input → that dataset. GPU T4, Internet on.
  3. Save Version → Save & Run All (Commit).
  4. Output tab → `edsr_pop.zip` → copy `edsr_best.pth` to `InfraVision/checkpoints/sr/edsr_best.pth`. The last cell says whether it beats bicubic. If it doesn't, set `sr.backend: bicubic`.
- [x] **U9: train detector v2 on Kaggle (POP + flood v2 + AIResQ + WiSARD)**
  - Done 2026-11-01: installed `checkpoints/yolo/sar_v2.pt` (now first in `detection.weights`). Test mAP50 POP 0.620 → 0.614, AIResQ 0.147 → 0.396, WiSARD 0.268 → 0.630, empty flood blobs 26.8% → 2.7%. Best val epoch 2 (POP-heavy val), early stop at 10. Details in `docs/RESULTS.md`.
  1. Kaggle → Datasets → New Dataset → upload `InfraVision/human_detection/sar_yolo.zip` (1.93 GB, private).
  2. Code → New Notebook → File → Import Notebook → `colab_training/train_sar_yolo_kaggle.ipynb`. Add Input → that dataset. Accelerator GPU T4 x2, Internet On.
  3. Save Version → Save & Run All (Commit). ~3–4 h (30 epochs max, early stop).
  4. Output tab → `sar_1024.zip` → put it in `datasets/outputs/`. Claude installs it as `checkpoints/yolo/sar_v2.pt` (the old `best.pt` stays) and compares.
  - The notebook prints before/after mAP50 on test_pop, test_flood, test_airesq, test_wisard and the empty-blob rate (gate < 2%).
- [x] **U10: train detector v3 on Kaggle** (after U9 finishes)
  - Done 2026-11-01: `checkpoints/yolo/sar_v3.pt` (dashboard default). HIT-UAV 0.344 → 0.851, LLVIP 0.044 → 0.705, RGBT 0.102 → 0.327, AIResQ 0.396 → 0.457, WiSARD 0.630 → 0.637, **POP 0.614 → 0.525**. Details in `docs/RESULTS.md`.
  0. Ready 2026-11-01: `start.pt` = v2 at `human_detection/sar_v2_start/start.pt`; zip rebuilt with balanced val (every 4th POP val frame: 1372 val) and the notebook fine-tunes with SGD lr 0.002 cosine, patience 8.
  1. (done) Put U9's `sar_1024.zip` in `datasets/outputs/`. Claude runs `scripts/install_detector.py ../datasets/outputs/sar_1024.zip --name sar_v2 --sweep` → installs `checkpoints/yolo/sar_v2.pt` (old `best.pt` kept), prints before/after, writes `human_detection/sar_v2_start/start.pt`.
  2. **Faster upload (use this):** Kaggle → two private datasets: `human_detection/sar_v3_addon.zip` (0.69 GB: only the data the v2 upload lacks) and `human_detection/sar_v2_start/start.pt` (19 MB).
  3. Import `colab_training/train_sar_v3_addon_kaggle.ipynb`, Add Input → **three**: the v2 `sar_yolo` dataset (already on Kaggle), the add-on, `start.pt`. GPU T4 x2, Internet On, Save & Run All. Same training data as the full zip (dry-run tested with 3 inputs).
     (Full alternative: `sar_v3_yolo.zip` (1.86 GB) + `train_sar_v3_yolo_kaggle.ipynb` with 2 inputs.)
  4. The log prints min/epoch and an ETA after every epoch. Output `sar_v3.zip` → `datasets/outputs/`.
  - Data: POP (every 3rd) + flood v2, AIResQ, WiSARD, **RGBTDronePerson** (every 3rd), **HIT-UAV** and **LLVIP persons** (every 8th; large close oblique people): 12.2k train frames; 7 test sets incl. `test_llvip`. 4 data workers, batch 16, 20 epochs, patience 6. Dry-run tested on CPU (before LLVIP was added; same code path).
  - Baselines of the current model: HIT-UAV test mAP50 0.262, RGBTDronePerson val 0.055 @640 / 0.076 @1024.
- [x] **U7: disaster datasets** (replaces U3)
  - Done 2026-11-01: WiSARD full (browser download, zip integrity-tested) + AIResQ benchmark; both in `sar_yolo.zip`. Also **HIT-UAV** (Zenodo, md5 ok) and **RGBTDronePerson** (Google Drive, CC BY 4.0, thermal tiny people) → converters in `human_detection/datasets/convert_*.py`, used in U10. Checked and rejected: SeaDronesSee (RGB/NIR, not thermal), Figshare SAR_Drone (260 whole-image labels, simulated), VisDrone/HERIDAL/C2A (RGB). Still missing: real aerial thermal people *in* water (WiSARD/AIResQ have people at lakeshores and rivers, not swimming).
  - HIT-UAV (thermal drone, person boxes, CC BY 4.0): `git clone https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset` into `datasets/HIT-UAV/`.
  - WiSARD (wilderness SAR, thermal + RGB, MIT-style licence), from sites.google.com/uw.edu/wisard: first the **971.6 MB Multi Modal Sample** into `datasets/WiSARD/`. Only get the 40.5 GB full set if the sample shows a clear gain.
  - Then hand C8 to Claude.
  - 2026-11-01: **AIResQ benchmark** downloaded instead (Zenodo 17405074, CC BY 4.0, 1989 thermal drone images 640x512, 2643 person boxes, YOLO format) at `datasets/AIResQ/Benchmark/{images,labels}`. The full 2048x1536 AIResQ (Zenodo 17405323) is access-restricted (request on Zenodo). HIT-UAV not cloned yet. WiSARD full (40 GB) on hold: no water scenes; the user will send the link in a new chat; test server speed first, use `scripts/pdl.sh` if slow.
- [ ] **U8: real flood / water-rescue frames** (now the top priority)
  - Thermal clips of people in water (rescue / news / YouTube), top-down **and** oblique. Internal testing only, do not redistribute. Also send the IWT-YOLO dataset request email.
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
- [ ] **C1 v2: fix the flood shortcut** (found 2026-10-31)
  - Data done 2026-11-01: `prepare_flood_aug.py` adds one empty decoy blob per flooded person (7343 decoys), saved in `flood_decoys.json`; `scripts/eval_flood_decoys.py` measures the rate (current model 26.8%, gate < 2%). Retrain = U9.
  - Add the same number of empty water blobs (no person) to every flooded copy, train and test, so a blob no longer means "person". Rebuild the zip; then one more YOLO run on Kaggle. Gate: empty blobs detected < 2%, and the flood test score drops to a believable value.
- [ ] **C6: detection evaluation** (needs U1)
  - 2026-11-01: `scripts/install_detector.py --sweep` runs the pipeline Detector (tiling included) old vs new weights at imgsz x tiles on POP test, POP flood, HIT-UAV test, RGBTDronePerson val (100 frames each, ~1 h CPU) → `results/detection_<name>_sweep.json`. Run it when U9 lands.
  - Sweep imgsz, tiling (`detection.tile`) and TTA on POP val, then pick the best.
  - Report test-set P / R / mAP@0.5 to `results/detection_upgrade.json` and update `docs/RESULTS.md`.
- [ ] **C8: disaster datasets → YOLO + eval** (needs U7)
  - 2026-11-01: `human_detection/datasets/make_sar_zip.py` builds `sar_yolo.zip`: POP (every 2nd frame) + flood v2, AIResQ split by flight (4 test / 2 val / 11 train flights; train adds 2x2 tiles since people are 7x9 px), WiSARD split by sequence (test: MtErie_IR_0004, Karen_FLIR_IR_5, FHL_IR_0567, Airfield_FLIR_IR_2; val: Carnation_FLIR_IR_2, FHL_IR_0406). `train.txt` 11.6k balanced entries, `train_all.txt` 15.2k. Notebook `colab_training/train_sar_yolo_kaggle.ipynb` dry-run tested end to end on CPU. Left: install + evaluate the U9 result.
  - Partial 2026-11-01: WiSARD IR sample (264 frames) mAP50 0.435 @640 / 0.411 @1024 → `results/disaster_eval.json`. **AIResQ eval was running when the session ended; rerun:** `ultralytics val` of `checkpoints/yolo/best.pt` on `datasets/AIResQ/Benchmark` (data yaml: path=that folder, train/val=images, names {0: person}; imgsz 640 and 1024) and add it to `disaster_eval.json`.
  - Converters for HIT-UAV and WiSARD into YOLO format (person only, full resolution), plus WiSARD thermal/RGB pairs for the colorizer.
  - Evaluate the POP YOLO on POP test, `test_flood`, HIT-UAV and WiSARD → `results/disaster_eval.json`. Then a combined Kaggle fine-tune.
- [ ] **C9: water data tooling** (after U8 clips arrive)
  - Script: video → frames (every N s, dedupe) → YOLO pre-labels in YOLO txt for fixing in CVAT/Roboflow → `datasets/flood_real/`.
  - Copy-paste augmentation: real thermal people (POP crops, cut at a water line) pasted into real thermal water backgrounds from the clips; replaces the painted blobs of C1.
  - Check: tiling on small frames (600x460 web image with 3x3 tiles magnifies people ~4x); auto-disable tiles when tiles get smaller than ~400 px if it helps.
  - Then one Kaggle YOLO run on POP + AIResQ (+ HIT-UAV) + water data; gate on the real water frames.
- [ ] **C7: dashboard** (RUN_ORDER 4)
  - 2026-11-01 (pm): **Video tab** done: upload a thermal clip, sample one frame every N s (max frames), detection-only (~0.7 s/frame CPU) or full pipeline (~1–2 s/frame); annotated WebM (VP8, plays in the browser without ffmpeg), people-over-time chart, CSV. Tested through the Gradio API on a WiSARD clip. Left: a look in a real browser.
  - 2026-11-01 (pm): the LLVIP colorizer alone gave a near-black "night photo" (stage 3 brightness ~25 of 255 vs ~110 thermal), hiding people. New `color_post.view` (default `luma`: thermal brightness + AI colour tint; `palette`: inferno; `ai`: old view) and app controls "Colour view" + "Contrast boost after SR" (default CLAHE clip 1.5 → 1.0: stronger CLAHE flattens people into foliage). Before/after: `results/display_before_after.jpg`. Detection is unaffected (runs on the original frame).
  - 2026-11-01: bicubic default (EDSR 61 s/frame CPU), black-hot palette switch.
  - Mostly done 2026-10-31: `app.py` has the 4-stage view, confidence / tiling / hotspot-blend sliders, SR selector (EDSR / bicubic / off), victim-count badge, per-branch ms table, model quality from the checkpoints, downloads (overlay PNG; zip with all stages + JSON report) and a Batch tab (gallery, CSV, zip). Tested end to end through the Gradio API (stock yolo11s, CPU test colorizer): ~1.8 s per POP frame with bicubic. `--yolo` flag for testing before the POP weights land. **Left:** video / frame-stream mode, and a visual check in a browser (no Playwright here).
  - Extend `app.py`: 4-stage view, conf / tiling / hotspot sliders, SR selector, victim count, per-branch ms, export.
  - Check it with the Playwright MCP.
