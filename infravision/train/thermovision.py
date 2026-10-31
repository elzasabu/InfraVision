"""Train the ThermoVision U-Net colorizer (thermal -> RGB) on LLVIP pairs; keep the best by val PSNR.

    python scripts/pack_llvip.py                                     # laptop: datasets/LLVIP -> llvip_colorizer.zip
    python -m infravision.train.thermovision --data <llvip dir> --amp --resume
    python -m infravision.train.thermovision --smoke --device cpu    # 2 tiny epochs

`--data` takes either the packed zip layout ({train,test}/{ir,rgb}/x.jpg) or the
original LLVIP layout ({infrared,visible}/{train,test}/x.jpg).

Model contract (infravision/models/thermovision_colorizer.py): the whole raw
thermal frame resized to size x size, [0, 1] in; sigmoid RGB [0, 1] out.
Loss = L1 + edge (Sobel on luma) + VGG16 perceptual + chroma (L1 on Cb/Cr,
against the washed-out colour of a plain L1 regression). The encoder starts
from ImageNet ResNet34 (conv1 summed to one channel); its BatchNorm adapts to thermal
images (frozen ImageNet statistics include near-zero variances that blow small weight
changes up past the fp16 range).
Writes <out>/thermovision_best.pth (loadable by infravision.colorizer) and
metrics.json; val saturation is logged next to PSNR / SSIM.
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from ..config import find_dir, load_config
from ..io import IMAGE_EXTS
from ..metrics import psnr, ssim
from ..models.thermovision_colorizer import ThermoVisionUNet
from .colorizer import VGGPerceptual
from .common import CHECKPOINTS, BestKeeper, LossWatch, Resumer, add_common_args, autocast, fail, setup


def llvip_pairs(root, split):
    """(thermal path, RGB path) for one split of a packed or original LLVIP folder."""
    root = Path(root)
    packed = root / split / "ir"
    t, v = (packed, root / split / "rgb") if packed.is_dir() else (root / "infrared" / split, root / "visible" / split)
    names = sorted(p.name for p in t.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [(t / n, v / n) for n in names if (v / n).is_file()]


class LLVIPColor(Dataset):
    """-> (thermal 1xSxS, RGB 3xSxS), both in [0, 1]; whole frame (train: random 80-100% crop + flip)."""

    def __init__(self, pairs, size=256, train=True):
        if not pairs:
            raise ValueError("LLVIPColor: no pairs found")
        self.pairs, self.size, self.train = pairs, size, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tp, vp = self.pairs[i]
        t = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        v = cv2.cvtColor(cv2.imread(str(vp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if t is None or v is None:
            raise OSError(f"unreadable pair: {tp}")
        if v.shape[:2] != t.shape:
            v = cv2.resize(v, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
        if self.train:
            h, w = t.shape
            s = random.uniform(0.8, 1.0)
            ch, cw = round(h * s), round(w * s)
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
            t, v = t[y:y + ch, x:x + cw], v[y:y + ch, x:x + cw]
            if random.random() < 0.5:
                t, v = t[:, ::-1], v[:, ::-1]
        sz = (self.size, self.size)
        t = cv2.resize(np.ascontiguousarray(t), sz, interpolation=cv2.INTER_AREA)
        v = cv2.resize(np.ascontiguousarray(v), sz, interpolation=cv2.INTER_AREA)
        return (torch.from_numpy(t).float().div_(255)[None],
                torch.from_numpy(v).float().div_(255).permute(2, 0, 1))


def build_model(pretrained=True):
    m = ThermoVisionUNet()
    if pretrained:
        try:
            from torchvision.models import ResNet34_Weights, resnet34

            sd = {k: v for k, v in resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict().items()
                  if not k.startswith("fc.")}
            sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)   # RGB filters -> one grey channel
            m.encoder.load_state_dict(sd)
            print("encoder: ImageNet ResNet34")
        except Exception as e:  # no internet
            print(f"WARNING: ImageNet ResNet34 weights unavailable ({e.__class__.__name__}); encoder from scratch")
    return m


def luma(x):
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]


def chroma(x):
    r, g, b = x[:, :1], x[:, 1:2], x[:, 2:]
    return torch.cat([-0.168736 * r - 0.331264 * g + 0.5 * b, 0.5 * r - 0.418688 * g - 0.081312 * b], 1)


SOBEL = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def edges(x):
    k = torch.stack([SOBEL, SOBEL.T])[:, None].to(x)
    return F.conv2d(luma(x), k, padding=1)


def saturation(x):
    """Mean HSV saturation of RGB in [0, 1]."""
    mx, mn = x.amax(1), x.amin(1)
    return ((mx - mn) / (mx + 1e-6)).mean()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="LLVIP folder (default: configs datasets.llvip)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=256, help="model input side (multiple of 32)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--edge_weight", type=float, default=0.1)
    p.add_argument("--perceptual_weight", type=float, default=0.1)
    p.add_argument("--chroma_weight", type=float, default=1.0)
    p.add_argument("--no_pretrained", action="store_true", help="encoder from scratch (no ImageNet download)")
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_val", type=int, default=500, help="evenly spaced LLVIP test pairs scored each epoch")
    p.add_argument("--steps_per_epoch", type=int, default=0, help="0 = full pass")
    add_common_args(p, CHECKPOINTS / "colorizer")
    args = p.parse_args(argv)
    if args.smoke:
        args.epochs, args.max_train, args.max_val, args.batch, args.size = 2, 8, 4, 2, 64
        args.steps_per_epoch, args.perceptual_weight, args.no_pretrained = 3, 0.0, True
    device, amp, out = setup(args)
    random.seed(args.seed)

    data = args.data or find_dir(load_config()["datasets"]["llvip"])
    if data is None:
        raise SystemExit("LLVIP not found: pass --data or set datasets.llvip in configs/infravision.yaml")
    train_pairs, val_pairs = llvip_pairs(data, "train"), llvip_pairs(data, "test")
    random.Random(args.seed).shuffle(train_pairs)
    if args.max_train:
        train_pairs = train_pairs[: args.max_train]
    val_pairs = val_pairs[:: max(1, len(val_pairs) // args.max_val)][: args.max_val]
    print(f"data {data}: train pairs {len(train_pairs)}, val pairs {len(val_pairs)}")

    tl = DataLoader(LLVIPColor(train_pairs, args.size, True), args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(LLVIPColor(val_pairs, args.size, False), args.batch, shuffle=False, num_workers=args.workers)

    G = build_model(not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    perc = None
    if args.perceptual_weight > 0:
        try:
            perc = VGGPerceptual().to(device)
        except Exception as e:  # no internet for the VGG weights
            print(f"WARNING: VGG16 weights unavailable ({e.__class__.__name__}); perceptual loss disabled")

    meta = dict(model="ThermoVisionUNet", architecture="pretrained_unet", encoder="resnet34", input_size=args.size,
                trained_on="LLVIP", perceptual=perc is not None)
    keeper = BestKeeper(out, "thermovision", "val_psnr", meta)
    resumer = Resumer(out, keeper, args.resume, G=G, opt=opt, sched=sched, scaler=scaler)

    watch = LossWatch()
    for epoch in range(resumer.start_epoch(), args.epochs + 1):
        G.train()
        tot = n = 0
        for step, (t, v) in enumerate(tl, 1):
            t, v = t.to(device, non_blocking=True), v.to(device, non_blocking=True)
            with autocast(device, amp):
                fake = G(t)
            fake = fake.float()
            loss = (F.l1_loss(fake, v) + args.edge_weight * F.l1_loss(edges(fake), edges(v))
                    + args.chroma_weight * F.l1_loss(chroma(fake), chroma(v)))
            if perc is not None:
                with autocast(device, amp):
                    loss = loss + args.perceptual_weight * perc(fake * 2 - 1, v * 2 - 1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            watch(loss.item())
            tot, n = tot + loss.item(), n + 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(tl)} loss {tot / n:.4f}", flush=True)
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
        sched.step()

        G.eval()
        ps, ss, sat, sat_t, con, con_t = [], [], [], [], [], []
        with torch.inference_mode():
            for i, (t, v) in enumerate(vl):
                with autocast(device, amp):
                    pred = G(t.to(device))
                pred = pred.float().cpu().clamp(0, 1)
                ps.append(psnr(pred, v))
                ss.append(ssim(pred, v))
                sat.append(saturation(pred))
                sat_t.append(saturation(v))
                con.append(luma(pred).flatten(1).std(1))
                con_t.append(luma(v).flatten(1).std(1))
                if i == 0:
                    save_image(torch.cat([t.repeat(1, 3, 1, 1), pred, v]), out / f"val_samples_epoch{epoch:03d}.png",
                               nrow=t.shape[0])
        m = dict(train_loss=tot / max(n, 1), val_psnr=torch.stack(ps).mean().item(),
                 val_ssim=torch.stack(ss).mean().item(), val_saturation=torch.stack(sat).mean().item(),
                 target_saturation=torch.stack(sat_t).mean().item(),
                 contrast_ratio=(torch.cat(con).mean() / torch.cat(con_t).mean().clamp_min(1e-6)).item())
        if not math.isfinite(m["val_psnr"]):   # the model is poisoned; more epochs cannot fix it
            fail(f"epoch {epoch}: val PSNR is NaN/inf ({m})")
        # A flat, average-colour output (run 1) has ~0.17 of the photo's contrast; a learning model is past 0.5.
        if epoch >= 2 and m["contrast_ratio"] < 0.3 and not args.smoke:   # smoke = 3 steps on noise
            fail(f"epoch {epoch}: output is nearly flat (contrast ratio {m['contrast_ratio']:.2f} < 0.3)")
        best = keeper.update(epoch, G.state_dict(), m, {k: str(v) for k, v in vars(args).items()})
        print(f"epoch {epoch}/{args.epochs} loss {m['train_loss']:.4f} val PSNR {m['val_psnr']:.2f} dB "
              f"SSIM {m['val_ssim']:.4f} saturation {m['val_saturation']:.3f} (real {m['target_saturation']:.3f}) "
              f"contrast {m['contrast_ratio']:.2f}"
              + ("  *best*" if best else ""), flush=True)
        resumer.end_epoch(epoch, args.epochs)
    resumer.finish()
    print("Best:", keeper.best, "->", out / "thermovision_best.pth")


if __name__ == "__main__":
    main()
