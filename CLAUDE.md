# CLAUDE.md — InfraVision Project Context & Engineering Rules

## 1. Project Overview
InfraVision is a multi-model thermal infrared (IR) enhancement, colorization, and human detection pipeline for disaster search-and-rescue (SAR).
- **Input:** Single-channel grayscale thermal images (256x256 cropped or 1280x1024 full drone frames).
- **Branch A (Enhancement & Colorization):** Pre-enhancement -> Super-Resolution (x2) -> Thermal-to-RGB Colorization.
- **Branch B (Detection):** YOLO11s human detection running in parallel on the thermal branch, with bounding boxes scaled (x2) and overlaid onto Branch A's output.

## 2. Current Bottlenecks & Known Bugs (Ground Truth)
1. **Hardware Execution:** Code currently defaults to CPU execution (~41.8s per 1280x1024 frame). Must be migrated to CUDA + FP16 (`torch.amp.autocast`).
2. **EDSR Super-Resolution Flaw:** Current EDSR (x2) checkpoint gets 34.24 dB PSNR on POP vs 47.89 dB for Bicubic Interpolation because it was under-trained on out-of-domain data and runs *after* aggressive CLAHE (which amplifies sensor noise).
3. **YOLO11s Aerial Domain Gap:** Trained only on FLIR ADAS (street-level pedestrians), resulting in 0.026 mAP@0.5 on POP aerial drone frames (where humans are ~40px tall). Needs POP fine-tuning + SAHI tiled inference.
4. **Colorizer Desaturation (ThermoVision exp7):** Paired ResNet34-UNet trained on street scenes outputs washed-out saturation (0.047) and grey-green haze on nature/flood scenes. Needs LAB chroma boosting and optional thermal-hotspot fusion so warm human bodies are never camouflaged.
5. **Missing GUI:** Needs an interactive web dashboard (Rescue Operator view + Admin metrics view) with side-by-side comparison, thermal hotspot overlay toggle, and downloadable results.

## 3. Coding & Optimization Standards
- **Device Management:** Always resolve device dynamically (`cuda` if available, fallback to `cpu`). Support FP16 inference on CUDA.
- **No Hardcoded Paths:** Centralize all dataset paths (`POP`, `LLVIP`, `FLIR`) and checkpoint weights in a single `config.yaml` or `src/config.py`.
- **Reproducibility:** Keep seed fixed (`seed = 0`) for all evaluation benchmarks and log metrics (PSNR, SSIM, LPIPS, Precision, Recall, mAP@0.5, and ms/image per branch) to JSON/CSV in `results/`.
- **Non-Destructive Refactoring:** Never overwrite existing `.pt` or `.pth` checkpoints. Save new weights with explicit version tags (e.g., `edsr_pop_ft_v2.pt`, `yolo11s_pop_flood_v2.pt`).

## 4. Verification Protocol
After every code modification, verify success by printing a structured status banner:
`[CAVE-PONY: RUN_ORDER STEP <N> VERIFIED | GPU: <USED/FREE> | LATENCY: <X>ms]`