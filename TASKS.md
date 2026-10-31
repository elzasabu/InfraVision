# TASKS.md: InfraVision Rev1 task board

**How to use:** start a new chat with "Do task C2 from TASKS.md". Each task stands on its own. When a task is finished, tick it, add a one-line result under it, and commit.
Owner **U** = the user (accounts, downloads, uploads, pushing). Owner **C** = Claude (code, local runs).

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

## User tasks

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
- [ ] **U4: push when asked:** `git push -u origin feature/rev1` (from `InfraVision/`).

## Claude tasks

- [x] **C0: Phase 0.** FP16 autocast, `scripts/benchmark_baseline.py`, merged the ThermoVision branch, dataset paths in config, POP → YOLO zip for Kaggle.
- [ ] **C1: flood-occlusion augmentation** (RUN_ORDER 2.2)
  - `scripts/prepare_flood_aug.py`: offline copies of ~30% of POP train frames.
  - In each copy, the lower 30–60% of every person box is filled with cool, water-like intensity plus ripple noise. Labels stay unchanged.
  - Add the copies to `human_detection/datasets/pop`, rebuild the zip, and build a flood-occluded test set from POP test.
  - Done when: the augmented images look right by eye (save a grid) and `make_pop_zip.py` includes them.
- [ ] **C2: Branch A reorder** (RUN_ORDER 1.1–1.2)
  - New order: `cv2.bilateralFilter` → SR → CLAHE (clip 1.5, 8×8) after SR.
  - Config switch `sr.backend: bicubic | edsr`.
  - Removes the 2.5 s NLMeans step. Tests pass. Update the defaults in `configs/infravision.yaml`.
- [ ] **C3: prepare EDSR training on POP** (RUN_ORDER 1.3)
  - Check `infravision/train/sr.py` against the recipe: 25 epochs, batch 16, AdamW 1e-4, cosine schedule, loss 0.9·L1 + 0.1·(1−SSIM).
  - Make a Kaggle notebook plus a data zip. Smoke-run 1 epoch on CPU.
  - Then **U** runs it on Kaggle. Gate: beats bicubic on POP val (`python -m infravision.evaluate --skip colorizer grid timing`).
- [ ] **C4: colorizer post-processing** (RUN_ORDER 3), CPU only
  - LAB chroma boost (`chroma_boost: 1.8`) and thermal hotspot fusion (top 5% heat, magma LUT, `alpha: 0.35`).
  - Both are config switches, with tests.
- [ ] **C5: colorizer training** (needs U2)
  - Add ThermoVision U-Net training (L1 + edge + VGG + Lab-chroma losses) to `infravision/train/colorizer.py`, plus a Kaggle notebook.
- [ ] **C6: detection evaluation** (needs U1)
  - Sweep imgsz, tiling (`detection.tile`) and TTA on POP val, then pick the best.
  - Report test-set P / R / mAP@0.5 to `results/detection_upgrade.json` and update `docs/RESULTS.md`.
- [ ] **C7: dashboard** (RUN_ORDER 4)
  - Extend `app.py`: 4-stage view, conf / tiling / hotspot sliders, SR selector, victim count, per-branch ms, export.
  - Check it with the Playwright MCP.
