"""Shared bits for the training scripts: device/AMP setup, best-checkpoint saving."""
import json
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
