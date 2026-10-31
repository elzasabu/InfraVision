"""Shared bits for the training scripts: device/AMP setup, best-checkpoint saving."""
import json
import math
import os
import time
from pathlib import Path

import torch

from ..config import ROOT, pick_device

CHECKPOINTS = ROOT / "checkpoints"


def add_common_args(p, out_default):
    p.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda, cuda:0")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only; ignored on CPU)")
    p.add_argument("--out", default=None,
                   help=f"folder for best checkpoint + metrics.json (default {Path(out_default).relative_to(ROOT)}, "
                        "or checkpoints/smoke/... with --smoke so smoke runs never replace real weights)")
    p.set_defaults(out_default=str(out_default))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (few images, few steps) to check the code path on a CPU")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/last.pth if it exists (saved after every epoch)")
    return p


def setup(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    if args.amp and not amp:
        print("--amp needs CUDA; training in full precision on", device)
    if args.out is None:
        d = Path(args.out_default)
        args.out = str(CHECKPOINTS / "smoke" / d.name if args.smoke else d)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} amp={amp} out={out}")
    return device, amp, out


def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


class BestKeeper:
    """Saves <name>_best.pth whenever the val metric improves, plus metrics.json."""

    def __init__(self, out, name, metric, meta, higher_is_better=True):
        self.out, self.name, self.metric, self.meta = Path(out), name, metric, meta
        self.sign = 1 if higher_is_better else -1
        self.best = None
        self.history = []
        self.t0 = time.time()

    def update(self, epoch, state_dict, metrics, extra_config=None):
        self.history.append(dict(epoch=epoch, **metrics))
        score = metrics[self.metric]
        improved = self.best is None or self.sign * score > self.sign * self.best[self.metric]
        if improved:
            self.best = dict(epoch=epoch, **metrics)
            torch.save({"state_dict": state_dict, "meta": self.meta, "metrics": self.best},
                       self.out / f"{self.name}_best.pth")
        with open(self.out / "metrics.json", "w") as f:
            json.dump(dict(best=self.best, selected_by=self.metric, history=self.history,
                           config=extra_config or {}, meta=self.meta,
                           minutes=round((time.time() - self.t0) / 60, 2)), f, indent=2)
        return improved


def atomic_torch_save(obj, path):
    """Write to a temp file, then rename, so a disconnect never leaves a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fail(msg):
    """Stop a broken run now instead of after hours of Kaggle GPU time."""
    raise SystemExit(f"HEALTH CHECK FAILED: {msg}")


class LossWatch:
    """Stops training after `limit` NaN/inf losses in a row (a few are normal with AMP)."""

    def __init__(self, limit=20):
        self.limit, self.bad = limit, 0

    def __call__(self, loss):
        self.bad = 0 if math.isfinite(loss) else self.bad + 1
        if self.bad >= self.limit:
            fail(f"{self.bad} training steps in a row gave a NaN/inf loss")


class Resumer:
    """Saves <out>/last.pth after every epoch and, with --resume, continues from it.

    `objs` are the things whose state must survive a disconnect (models,
    optimisers, schedulers, AMP scaler); None entries are ignored. The best
    checkpoint and the metrics history of `keeper` are restored too, so
    metrics.json keeps every epoch. A finished run writes <out>/DONE.
    """

    def __init__(self, out, keeper, resume, **objs):
        self.out, self.keeper, self.resume = Path(out), keeper, resume
        self.objs = {k: v for k, v in objs.items() if v is not None}
        self.last = self.out / "last.pth"
        self.t_epoch = time.time()

    def start_epoch(self):
        if not (self.resume and self.last.exists()):
            if self.resume:
                print(f"--resume: no {self.last} yet, starting from epoch 1")
            return 1
        ck = torch.load(self.last, map_location="cpu", weights_only=False)
        for k, v in self.objs.items():
            if k in ck["state"]:
                v.load_state_dict(ck["state"][k])
        self.keeper.best, self.keeper.history = ck["best"], ck["history"]
        print(f"resumed from {self.last}: epoch {ck['epoch']} done, best so far {self.keeper.best}")
        return ck["epoch"] + 1

    def end_epoch(self, epoch, epochs):
        atomic_torch_save(dict(epoch=epoch, state={k: v.state_dict() for k, v in self.objs.items()},
                               best=self.keeper.best, history=self.keeper.history), self.last)
        dt = time.time() - self.t_epoch
        self.t_epoch = time.time()
        left = (epochs - epoch) * dt / 60
        print(f"  epoch {epoch} took {dt / 60:.1f} min; about {left:.0f} min left; saved {self.last.name}", flush=True)

    def finish(self):
        (self.out / "DONE").write_text(json.dumps(dict(best=self.keeper.best), indent=2))
