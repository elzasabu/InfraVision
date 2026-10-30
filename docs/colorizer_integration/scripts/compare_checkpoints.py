"""Step 1 of the colorizer integration: same inputs, same metrics for every candidate colorizer.

Candidates: ThermoVision exp3 / exp4 (ThermalColorizationNet, needs the ThermoVision source folder), the
ThermoVision U-Nets exp6 / exp6b / exp7 best / exp7 last (through infravision.colorizer.ColorizerAdapter,
checked against ThermoVision's own unet_models.load_unet) and the current InfraVision "tiny" colorizer.

Identical preprocessing for all: grey frame -> 256 x 256 (cv2 INTER_AREA) -> each model's own value range
([0,1] for ThermoVision, [-1,1] for the InfraVision ResnetGenerator). Target RGB -> 256 x 256 (INTER_AREA).
"tiny (pipeline)" is the tiny model as the pipeline runs it (CLAHE + denoise, longer side 640), for reference.

usage (read-only on every input folder; writes only to --out):
    python -B compare_checkpoints.py --tv <ThermoVision dir> --tiny <colorizer_best.pth> \
        --data data/paired_disaster --pop POP/test/JPEGImages --out docs/colorizer_integration
    python -B compare_checkpoints.py ... --time-only exp7_best      # one timing run (own process, peak RAM)
"""
import argparse
import csv
import ctypes
import glob
import json
import os
import subprocess
import sys
import time
import types
from ctypes import wintypes

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

p = argparse.ArgumentParser()
p.add_argument("--tv", required=True)
p.add_argument("--tiny", required=True)
p.add_argument("--data", required=True)
p.add_argument("--pop", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=50, help="pairs per paired set (LLVIP test, FLIR val)")
p.add_argument("--threads", type=int, default=12)
p.add_argument("--time-only", default=None)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(0)

sys.path.insert(0, REPO)
try:
    import yaml  # noqa: F401
except ImportError:  # the ThermoVision venv has no PyYAML; infravision falls back to its DEFAULTS
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda f: {})
from infravision.colorizer import ColorizerAdapter  # noqa: E402
from infravision.config import DEFAULTS  # noqa: E402
from infravision.enhance import load_colorizer, pre_enhance  # noqa: E402

sys.path.insert(0, a.tv)

CANDIDATES = {
    "exp3": ("tv_net", "checkpoints/exp3_chroma_best.pth"),
    "exp4": ("tv_net", "checkpoints/exp4_edge_magnitude_best.pth"),
    "exp6": ("unet", "experiments/exp6_resnet34_unet/best.pth"),
    "exp6b": ("unet", "experiments/exp6b_resnet34_random_init/best.pth"),
    "exp7_best": ("unet", "experiments/exp7_resnet34_unet_continue/best.pth"),
    "exp7_last": ("unet", "experiments/exp7_resnet34_unet_continue/last.pth"),
    "tiny": ("tiny", None),
    "tiny_pipeline": ("tiny_native", None),
}
S = 256


def peak_rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return c.PeakWorkingSetSize / 2 ** 20


def resize_gray(g):
    return cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA if max(g.shape) > S else cv2.INTER_LINEAR)


def load(name):
    kind, rel = CANDIDATES[name]
    if kind == "tv_net":
        import checkpoint_utils
        m, _ = checkpoint_utils.load_model(os.path.join(a.tv, rel))
        m.eval()

        @torch.inference_mode()
        def run(g):
            x = torch.from_numpy(resize_gray(g).astype(np.float32) / 255.0)[None, None]
            return m(x).clamp(0, 1)[0].float()
        return run, m
    if kind == "unet":
        ad = ColorizerAdapter(os.path.join(a.tv, rel), device="cpu", verbose=False)

        @torch.inference_mode()
        def run(g):
            x = torch.from_numpy(ad._input(g))[None, None]
            return ad.model(x).clamp(0, 1)[0].float()
        return run, ad
    G, _ = load_colorizer(a.tiny)
    pre = DEFAULTS["pre_enhance"]
    if kind == "tiny":
        @torch.inference_mode()
        def run(g):
            x = torch.from_numpy(resize_gray(g)).float().div(127.5).sub(1)[None, None]
            return ((G(x).clamp(-1, 1) + 1) / 2)[0].float()
        return run, G

    @torch.inference_mode()
    def run(g, work_side=640):
        h, w = g.shape
        s = work_side / max(h, w)
        gg = cv2.resize(g, (round(w * s) // 4 * 4, round(h * s) // 4 * 4), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(pre_enhance(gg, **pre)).float().div(127.5).sub(1)[None, None]
        y = (G(x).clamp(-1, 1) + 1) / 2
        return F.interpolate(y, size=(S, S), mode="bilinear", align_corners=False, antialias=True)[0].float()
    return run, G


def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def read_rgb_t(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return torch.from_numpy(cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255).permute(2, 0, 1)


def evenly(seq, n):
    if len(seq) <= n:
        return list(seq)
    return [seq[i] for i in np.linspace(0, len(seq) - 1, n).round().astype(int)]


# ---------------------------------------------------------------- data (fixed, evenly spaced, sorted ids)
rows = list(csv.DictReader(open(os.path.join(a.data, "manifest.csv"), newline="")))
test_rows = {}
for r in rows:
    if r["split"] == "test":
        test_rows.setdefault(r["id"], r)
llvip = [(i, os.path.join(a.data, test_rows[i]["thermal"].replace("\\", "/")),
          os.path.join(a.data, test_rows[i]["rgb"].replace("\\", "/"))) for i in evenly(sorted(test_rows), a.n)]
flir_ids = [ln.strip().split("_PreviewData")[0] for ln in open(os.path.join(a.tv, "dataset/align/align_validation.txt")) if ln.strip()]
flir = [(i, os.path.join(a.tv, "dataset/align/JPEGImages", i + "_PreviewData.jpeg"),
         os.path.join(a.tv, "dataset/align/JPEGImages", i + "_RGB.jpg")) for i in evenly(flir_ids, a.n)]
pop = evenly(sorted(glob.glob(os.path.join(a.pop, "*.JPG")) + glob.glob(os.path.join(a.pop, "*.jpg"))), 20)

# ---------------------------------------------------------------- timing mode (one candidate per process)
if a.time_only:
    base = peak_rss_mb()
    run, _ = load(a.time_only)
    frames = [read_gray(t) for _, t, _ in flir[:15] + llvip[:15]]
    ts = []
    for k, g in enumerate(frames):
        t0 = time.perf_counter(); run(g); dt = time.perf_counter() - t0
        if k >= 3:
            ts.append(dt)
    print(json.dumps(dict(name=a.time_only, n=len(ts), mean_s=float(np.mean(ts)), median_s=float(np.median(ts)),
                          peak_ram_mb=peak_rss_mb(), ram_before_load_mb=base, threads=a.threads)))
    sys.exit(0)

# ---------------------------------------------------------------- metrics
os.makedirs(a.out, exist_ok=True)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval()


def gauss(size=11, sigma=1.5, c=3):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(c, 1, size, size).contiguous()


W3 = gauss()


def ssim(x, y):
    x, y = x[None], y[None]
    mx, my = F.conv2d(x, W3, groups=3), F.conv2d(y, W3, groups=3)
    sxx = F.conv2d(x * x, W3, groups=3) - mx ** 2
    syy = F.conv2d(y * y, W3, groups=3) - my ** 2
    sxy = F.conv2d(x * y, W3, groups=3) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean().item()


def psnr(x, y):
    return (10 * torch.log10(1 / F.mse_loss(x, y).clamp_min(1e-10))).item()


def luma(x):
    return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]


def saturation(x):
    """Mean HSV saturation (0 = grey)."""
    mx, mn = x.max(0).values, x.min(0).values
    return ((mx - mn) / mx.clamp_min(1e-6)).mean().item()


def sobel_mag(img2d):
    k = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    t = img2d[None, None]
    gx, gy = F.conv2d(t, k.view(1, 1, 3, 3)), F.conv2d(t, k.t().reshape(1, 1, 3, 3))
    return (gx ** 2 + gy ** 2).sqrt()[0, 0]


def edge_corr(a2d, b2d):
    ea, eb = sobel_mag(a2d).flatten(), sobel_mag(b2d).flatten()
    return torch.corrcoef(torch.stack([ea, eb]))[0, 1].item()


models = {}
for name in CANDIDATES:
    print("loading", name, flush=True)
    models[name] = load(name)

# The vendored U-Net must reproduce ThermoVision's own code exactly.
import unet_models  # noqa: E402
ref, _ = unet_models.load_unet(os.path.join(a.tv, CANDIDATES["exp7_best"][1]))
vend = models["exp7_best"][1].model
vendor_diff = 0.0
with torch.inference_mode():
    for _, t, _ in flir[:5] + llvip[:5]:
        x = torch.from_numpy(resize_gray(read_gray(t)).astype(np.float32) / 255)[None, None]
        vendor_diff = max(vendor_diff, (ref(x) - vend(x)).abs().max().item())
# ThermoVision's own resize (PIL bilinear) vs the adapter's cv2 INTER_AREA, on exp7 outputs
pil_vs_area = []
with torch.inference_mode():
    for _, t, v in flir[:10] + llvip[:10]:
        with Image.open(t) as im:
            xp = torch.from_numpy(np.asarray(im.convert("L").resize((S, S), Image.BILINEAR), np.float32) / 255)[None, None]
        tgt = read_rgb_t(v)
        pil_vs_area.append(psnr(vend(xp).clamp(0, 1)[0], tgt) - psnr(models["exp7_best"][0](read_gray(t)), tgt))
print(f"vendored vs ThermoVision exp7 max |diff| = {vendor_diff:.2e}; PIL-bilinear minus INTER_AREA PSNR "
      f"= {np.mean(pil_vs_area):+.3f} dB", flush=True)

KEYS = ("psnr", "ssim", "lpips", "sat", "struct", "luma")
results, per_image, cache = {}, [], {}
for dname, items in (("llvip", llvip), ("flir", flir)):
    acc = {m: {k: [] for k in KEYS} for m in CANDIDATES}
    outs = {m: [] for m in CANDIDATES}
    tgts, tsat, tluma = [], [], []
    for iid, tp, vp in items:
        g = read_gray(tp)
        th = torch.from_numpy(resize_gray(g).astype(np.float32) / 255)
        tgt = read_rgb_t(vp)
        tgts.append(tgt); tsat.append(saturation(tgt)); tluma.append(luma(tgt).mean().item())
        row = dict(set=dname, id=iid)
        for m, (fn, _) in models.items():
            o = fn(g)
            outs[m].append(o); cache[(dname, iid, m)] = o
            A = acc[m]
            A["psnr"].append(psnr(o, tgt)); A["ssim"].append(ssim(o, tgt)); A["sat"].append(saturation(o))
            A["struct"].append(edge_corr(luma(o), th)); A["luma"].append(luma(o).mean().item())
            row[f"{m}_psnr"], row[f"{m}_ssim"] = round(A["psnr"][-1], 3), round(A["ssim"][-1], 4)
        per_image.append(row)
    T = torch.stack(tgts)
    res = dict(n=len(items), target_sat=float(np.mean(tsat)), target_luma=float(np.mean(tluma)))
    for m in CANDIDATES:
        O = torch.stack(outs[m])
        with torch.no_grad():
            lp = [lpips(O[i:i + 10], T[i:i + 10]).item() * len(O[i:i + 10]) for i in range(0, len(O), 10)]
        acc[m]["lpips"] = [sum(lp) / len(O)]
        res[m] = {k: float(np.mean(v)) for k, v in acc[m].items()}
        res[m]["_psnr_list"], res[m]["_ssim_list"] = acc[m]["psnr"], acc[m]["ssim"]
    results[dname] = res

pop_res = {m: {"sat": [], "struct": [], "luma": []} for m in CANDIDATES}
for path in pop:
    g = read_gray(path)
    th = torch.from_numpy(resize_gray(g).astype(np.float32) / 255)
    for m, (fn, _) in models.items():
        o = fn(g); cache[("pop", path, m)] = o
        pop_res[m]["sat"].append(saturation(o)); pop_res[m]["struct"].append(edge_corr(luma(o), th))
        pop_res[m]["luma"].append(luma(o).mean().item())
results["pop"] = {m: {k: float(np.mean(v)) for k, v in R.items()} for m, R in pop_res.items()}
results["pop"]["n"] = len(pop)

# combined paired set (100 images): LPIPS is averaged per set with equal set sizes
comb = {}
for m in CANDIDATES:
    L, Fl = results["llvip"][m], results["flir"][m]
    comb[m] = {k: (L[k] + Fl[k]) / 2 for k in KEYS}
results["combined"] = comb

# ---------------------------------------------------------------- timing + RAM (separate processes)
timing = {}
for m in CANDIDATES:
    cmd = [sys.executable, "-B", os.path.abspath(__file__), "--tv", a.tv, "--tiny", a.tiny, "--data", a.data,
           "--pop", a.pop, "--out", a.out, "--threads", str(a.threads), "--time-only", m]
    outp = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    timing[m] = json.loads(outp)
    print("timing", outp, flush=True)
results["timing"] = timing

# ---------------------------------------------------------------- selection rule
ranked = [m for m in CANDIDATES if m != "tiny_pipeline"]   # tiny_pipeline is a reference, not a different checkpoint
r_lp = {m: i + 1 for i, m in enumerate(sorted(ranked, key=lambda m: comb[m]["lpips"]))}
r_ss = {m: i + 1 for i, m in enumerate(sorted(ranked, key=lambda m: -comb[m]["ssim"]))}
order = sorted(ranked, key=lambda m: (r_lp[m] + r_ss[m], -comb[m]["struct"], timing[m]["median_s"]))
results["ranking"] = [dict(name=m, lpips_rank=r_lp[m], ssim_rank=r_ss[m], rank_sum=r_lp[m] + r_ss[m]) for m in order]

# paired bootstrap exp7_best vs the runner-up and vs exp3 on the 100 images
rng = np.random.default_rng(0)


def paired(m1, m2, key):
    d = np.array(results["llvip"][m1][f"_{key}_list"] + results["flir"][m1][f"_{key}_list"]) - \
        np.array(results["llvip"][m2][f"_{key}_list"] + results["flir"][m2][f"_{key}_list"])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((d > 0).mean())]


results["paired"] = {f"{order[0]}_minus_{m}": {k: paired(order[0], m, k) for k in ("psnr", "ssim")}
                     for m in order[1:] if m != order[0]}
results["checks"] = dict(vendored_vs_thermovision_max_abs_diff=vendor_diff,
                         pil_bilinear_minus_inter_area_psnr_db=float(np.mean(pil_vs_area)))
results["images"] = dict(llvip=[i for i, _, _ in llvip], flir=[i for i, _, _ in flir],
                         pop=[os.path.basename(x) for x in pop])
for d in ("llvip", "flir"):
    for m in CANDIDATES:
        results[d][m].pop("_psnr_list"); results[d][m].pop("_ssim_list")
json.dump(results, open(os.path.join(a.out, "checkpoint_comparison.json"), "w"), indent=1)
with open(os.path.join(a.out, "checkpoint_comparison_per_image.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_image[0])); w.writeheader(); w.writerows(per_image)

# ---------------------------------------------------------------- grids
T = 192
COLS = ["thermal"] + list(CANDIDATES) + ["real RGB"]
LABEL = {"exp7_best": "exp7 best", "exp7_last": "exp7 last", "tiny": "IV tiny", "tiny_pipeline": "IV tiny (pipeline)"}


def to_tile(t):
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).resize((T, T), Image.LANCZOS)


def grid(entries, name):
    img = Image.new("RGB", (T * len(COLS), (T + 18) * len(entries) + 20), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for k, c in enumerate(COLS):
        d.text((T * k + 5, 4), LABEL.get(c, c), fill=(0, 0, 0))
    for r, (dname, key, tpath, vpath) in enumerate(entries):
        y = 20 + r * (T + 18)
        g = read_gray(tpath)
        tiles = [Image.fromarray(resize_gray(g)).convert("RGB").resize((T, T))]
        tiles += [to_tile(cache[(dname, key, m)]) for m in CANDIDATES]
        tiles.append(to_tile(read_rgb_t(vpath)) if vpath else Image.new("RGB", (T, T), (60, 60, 60)))
        for k, t in enumerate(tiles):
            img.paste(t, (T * k, y))
        d.text((5, y + T + 2), f"{dname}: {os.path.basename(str(key))}" + ("" if vpath else "  (POP: no RGB)"), fill=(0, 0, 0))
    img.save(os.path.join(a.out, name))


grid([("llvip",) + (llvip[k][0], llvip[k][1], llvip[k][2]) for k in (5, 25, 45)] +
     [("flir",) + (flir[k][0], flir[k][1], flir[k][2]) for k in (10, 30, 40)], "grid_paired.png")
grid([("pop", pop[k], pop[k], None) for k in (2, 7, 12, 17)], "grid_pop.png")
print(json.dumps(dict(ranking=results["ranking"], checks=results["checks"]), indent=1))
print("done")
