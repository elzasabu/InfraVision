# Datasets and attribution

InfraVision does not redistribute any dataset. Download each one from its
owner, accept its licence, and place it as described in the README. The
licences below are summaries; the owner's terms are what apply.

| Dataset | Used for | Where it goes | Notes |
|---|---|---|---|
| **POP** (drone thermal, people) | Primary: SR training, YOLO training, all reported metrics, the demo | `POP/{train,val,test}/` | Thermal only (1280x1024), COCO person boxes. Split by flight: train F1-F12, val F13-F16, test T1-T3. Cite the POP authors as their release asks. |
| **LLVIP** (Jia et al., *LLVIP: A Visible-infrared Paired Dataset for Low-light Vision*, ICCV Workshops 2021) | Colorizer training (aligned thermal + RGB pairs), synthetic disaster set | source anywhere, built into `data/paired_disaster/` | Street-level, night. For academic research use. |
| **FLIR ADAS** (Teledyne FLIR free thermal dataset, aligned version) | Optional extra colorizer pairs; the earlier street-scene YOLO model | `human_detection/datasets/flir_align/` | Free for research under Teledyne FLIR's terms; not for commercial use. |
| **DIV2K** (Agustsson and Timofte, NTIRE 2017) | Only the original SR experiments in `SR/` | `SR/dataset/` | Research use. |

## Synthetic data

`data/paired_disaster/` is built from LLVIP by `paired_data/build_paired_dataset.py`.
The thermal and clean RGB frames are real LLVIP frames; the smoke, fog, dust and
darkness on the `rgb_degraded/` copies are **synthetic** (`paired_data/degrade.py`).
Anything measured on that set is labelled "synthetic" in `docs/RESULTS.md`. It does
not contain real floods, earthquakes or collapsed buildings.

## Optional, not downloaded

Drone search-and-rescue sets with both RGB and thermal, closer to the target use
than LLVIP, if you want a real three-panel demo on aerial views (ask first, they
are large):

- WiSARD (wilderness search and rescue, visible + thermal): https://sites.google.com/uw.edu/wisard/
- NII-CU Multispectral Aerial Person Detection (MAPD): https://www.nii-cu-multispectral.org/
