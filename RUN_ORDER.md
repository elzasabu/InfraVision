# RUN_ORDER.md — Phased Optimization & Upgrade Roadmap

Execute these phases strictly in order. Do not advance to the next phase until the verification gate for the current phase passes.

---

## PHASE 0: Repository Audit & GPU Baseline Profiling
**Goal:** Map the existing codebase, fix hardcoded CPU device calls, and record baseline metrics.
1. Scan the repository for all `.py` scripts, model definitions (EDSR, ThermoVision ResNet34-UNet, YOLO11s), and `.pt`/`.pth` checkpoints.
2. Build `scripts/benchmark_baseline.py` to measure:
   - GPU vs CPU runtime per branch (Pre-process, SR, Colorization, YOLO11s) on 10 POP test images (both 256x256 and 1280x1024).
   - Baseline PSNR/SSIM (SR), Saturation (Colorizer), and mAP@0.5 (YOLO11s on 100 POP test images).
3. Refactor all model loaders to use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and wrap inference in `torch.inference_mode()` + `torch.amp.autocast('cuda', dtype=torch.float16)`.
- **Verification Gate:** Pipeline runs end-to-end on GPU in <500ms per image with zero shape or dtype errors.

---

## PHASE 1: Fix Super-Resolution Quality & Preprocessing Order
**Goal:** Beat Bicubic Interpolation (>47.9 dB PSNR on POP) and eliminate noise amplification.
1. Reorder Branch A preprocessing:
   - Replace aggressive pre-SR CLAHE with mild bilateral denoising (`cv2.bilateralFilter`) on the raw thermal input.
   - Run Super-Resolution (`x2`) on clean normalized thermal data.
   - Apply gentle CLAHE (`clipLimit=1.5, tileGridSize=(8,8)`) on the luminance channel *after* SR.
2. Add a fallback switch allowing `bicubic` vs `edsr` vs `realesrgan-x2` in config.
3. Create `scripts/train_edsr_pop.py` to fine-tune the EDSR x2 model on POP thermal training crops (25 epochs, batch size 16, AdamW `lr=1e-4` with CosineAnnealingLR, loss `0.9 * L1 + 0.1 * (1 - SSIM)`).
- **Verification Gate:** Fine-tuned EDSR achieves higher PSNR and SSIM than Bicubic Interpolation on the 30 POP test images.

---

## PHASE 2: Upgrade Aerial & Flood Human Detection (YOLO11s + SAHI)
**Goal:** Raise POP drone human detection from 0.026 mAP@0.5 to >0.65 mAP@0.5 and handle partial flood submersion.
1. Integrate **SAHI** (Slicing Aided Hyper Inference) into the detection branch for frames >640px (`slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2`).
2. Create `scripts/prepare_flood_aug.py`:
   - Implement a custom Albumentations transform that simulates flood occlusion by masking the lower 30%–60% of person bounding boxes with cool background thermal intensity + Gaussian water ripple noise on 30% of training samples.
3. Fine-tune `yolo11s` on the POP dataset (`imgsz=640` or `1024`, 50 epochs, batch size 16, mosaic=1.0, scale=0.5) starting from the existing thermal checkpoint.
- **Verification Gate:** Evaluate on the 100 POP test images and synthetic flood-occluded test set; log Precision, Recall, and mAP@0.5 improvements to `results/detection_upgrade.json`.

---

## PHASE 3: Fix Disaster Colorization & Thermal Hotspot Preservation
**Goal:** Eliminate washed-out grey-green haze (0.047 saturation) and ensure warm human signatures stand out against floodwater/foliage.
1. In the Colorizer post-processing module:
   - Convert ResNet34-UNet RGB output to `CIE LAB` color space.
   - Apply adaptive chrominance scaling (`A` and `B` channels multiplied by configurable `chroma_boost=1.8`) and clahe on the `L` channel.
2. Implement **Thermal Hotspot Fusion (`alpha_blend_hotspots`)**:
   - Extract the top 5% intensity percentiles from the raw super-resolved thermal frame.
   - Apply an Ironbow/Magma false-color LUT strictly to high-temperature anomalies and alpha-blend (default `alpha=0.35`, toggleable) onto the colorized RGB image so submerged or occluded victims remain visually unmistakable.
- **Verification Gate:** Saturation increases from ~0.047 to >0.18 on test pairs while maintaining structural SSIM, and human heat signatures remain distinct on POP and water test frames.

---

## PHASE 4: Build the Rescue Monitoring Dashboard (GUI)
**Goal:** Deliver the centralized web application defined in the SRS/DFD Level 2.
1. Build `app.py` (Gradio / FastAPI web interface) supporting:
   - Single image upload, batch folder evaluation, and video/drone frame-stream mode.
   - **4-Stage View:** Raw Thermal | Super-Resolved Thermal | AI Colorized | Final Tactical Overlay (with YOLO11s boxes + confidence scores).
   - **Interactive Controls:** Confidence threshold slider, SAHI tiling toggle, Thermal Hotspot Blend slider (0% to 100%), and SR model selector.
   - **Telemetry & Export:** Victim count badge, per-branch latency breakdown (ms), PSNR/SSIM readout, and one-click report/image download.
2. Test UI rendering and responsiveness via Playwright MCP.
- **Verification Gate:** Web dashboard launches cleanly on `localhost`, processes a full POP frame in <250ms on GPU, and exports annotated rescue imagery.