"""Train the ThermoVision U-Net colorizer (thermal -> RGB) on LLVIP pairs; keep the best by val PSNR.

    python scripts/pack_llvip.py                                     # laptop: datasets/LLVIP -> llvip_colorizer.zip
    python -m infravision.train.thermovision --data <llvip dir> --amp --resume
    python -m infravision.train.thermovision --smoke --device cpu    # 2 tiny epochs
    # several paired sets, sampled by weight, best checkpoint by the first set's val PSNR:
    python -m infravision.train.thermovision --data cart=/d/cart:3 kaist=/d/kaist:1 --gan_weight 0.01

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

`--gan_weight` > 0 adds a conditional PatchGAN (LSGAN, spectral norm) from epoch `--gan_start`:
plain L1 regression averages plausible colours into grey-brown; the discriminator pushes the
output towards realistic colour statistics (pix2pix: L1 + ~0.01 adversarial).
"""
import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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

    def __init__(self, pairs, size=256, train=True, geo_aug=0.0, thermal_aug=False):
        if not pairs:
            raise ValueError("LLVIPColor: no pairs found")
        self.pairs, self.size, self.train = pairs, size, train
        # geo_aug: probability of a random 90-degree rotation / vertical flip of the pair. CART is
        # mostly oblique (sky on top), so without it the model paints "blue = top" on nadir frames.
        # thermal_aug: random gamma / contrast / noise / blur on the thermal input only (other cameras).
        self.geo_aug, self.thermal_aug = float(geo_aug), bool(thermal_aug)

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
        if self.train and self.geo_aug and random.random() < self.geo_aug:
            k = random.randint(0, 3)
            t, v = np.rot90(t, k), np.rot90(v, k)
            if random.random() < 0.5:
                t, v = t[::-1], v[::-1]
            t, v = np.ascontiguousarray(t), np.ascontiguousarray(v)
        tt = torch.from_numpy(t).float().div_(255)[None]
        if self.train and self.thermal_aug:
            lo, hi = np.percentile(t, (random.uniform(0, 3), random.uniform(97, 100)))
            tt = ((tt - lo / 255) / max((hi - lo) / 255, 1e-3)).clamp_(0, 1)          # camera AGC stretch
            tt = tt.pow(random.uniform(0.7, 1.4))                                      # gamma
            if random.random() < 0.3:
                k = random.choice((3, 5))
                tt = F.avg_pool2d(tt[None], k, 1, k // 2)[0]                           # mild blur
            tt = (tt + torch.randn_like(tt) * random.uniform(0, 0.03)).clamp_(0, 1)    # sensor noise
        return tt, torch.from_numpy(v).float().div_(255).permute(2, 0, 1)


def parse_sources(specs):
    """["cart=/d/cart:3", "/d/llvip"] -> [(name, path, weight)]. Plain path = weight 1."""
    out = []
    for i, s in enumerate(specs):
        name, _, rest = s.rpartition("=") if "=" in s.split("/")[0] else ("", "", s)
        path, w = rest, 1.0
        if ":" in rest and rest.rsplit(":", 1)[1].replace(".", "", 1).isdigit():
            path, w = rest.rsplit(":", 1)[0], float(rest.rsplit(":", 1)[1])
        out.append((name or Path(path).name or f"set{i}", path, w))
    return out


class PatchD(torch.nn.Module):
    """Conditional PatchGAN on (thermal, RGB): 70x70 receptive field, spectral norm, no BatchNorm."""

    def __init__(self, ch=64):
        super().__init__()
        sn = torch.nn.utils.spectral_norm
        layers, c_in = [], 4
        for i, c in enumerate((ch, ch * 2, ch * 4, ch * 8)):
            layers += [sn(torch.nn.Conv2d(c_in, c, 4, 2 if i < 3 else 1, 1)), torch.nn.LeakyReLU(0.2, True)]
            c_in = c
        layers.append(sn(torch.nn.Conv2d(c_in, 1, 4, 1, 1)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t, rgb):
        return self.net(torch.cat([t * 2 - 1, rgb * 2 - 1], 1))


def build_model(pretrained=True, reverse=False):
    m = ThermoVisionUNet(in_ch=3, out_ch=1) if reverse else ThermoVisionUNet()
    if pretrained:
        try:
            from torchvision.models import ResNet34_Weights, resnet34

            sd = {k: v for k, v in resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict().items()
                  if not k.startswith("fc.")}
            if not reverse:
                sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)   # RGB filters -> one grey channel
            m.encoder.load_state_dict(sd)
            print("encoder: ImageNet ResNet34")
        except Exception as e:  # no internet
            print(f"WARNING: ImageNet ResNet34 weights unavailable ({e.__class__.__name__}); encoder from scratch")
    return m


def luma(x):
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]


def rgb3(x):
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def chroma(x):
    r, g, b = x[:, :1], x[:, 1:2], x[:, 2:]
    return torch.cat([-0.168736 * r - 0.331264 * g + 0.5 * b, 0.5 * r - 0.418688 * g - 0.081312 * b], 1)


SOBEL = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def edges(x):
    k = torch.stack([SOBEL, SOBEL.T])[:, None].to(x)
    return F.conv2d(luma(x), k, padding=1)


def saturation(x):
    """Mean HSV saturation of RGB in [0, 1] (0 for a 1-channel image)."""
    if x.shape[1] == 1:
        return x.new_zeros(())
    mx, mn = x.amax(1), x.amin(1)
    return ((mx - mn) / (mx + 1e-6)).mean()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=None,
                   help="paired folders ({train,test}/{ir,rgb}), optionally name=path:weight; the first is the "
                        "primary set (best checkpoint by its val PSNR). Default: configs datasets.llvip")
    p.add_argument("--gan_weight", type=float, default=0.0, help="conditional PatchGAN loss weight (0 = off)")
    p.add_argument("--gan_start", type=int, default=2, help="first epoch with the GAN loss")
    p.add_argument("--geo_aug", type=float, default=0.0,
                   help="probability of a random 90-degree rotation / flip of a training pair (nadir views)")
    p.add_argument("--thermal_aug", action="store_true",
                   help="random contrast / gamma / blur / noise on the thermal input (other cameras)")
    p.add_argument("--reverse", action="store_true",
                   help="train the reverse model RGB -> thermal (rgb2thermal_best.pth), for pseudo-paired data")
    p.add_argument("--init", default="", help="start G from these weights (e.g. colorizer v2 for a fine-tune); "
                                             "not a resume: optimiser and epochs start fresh")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=256, help="model input side (multiple of 32)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--edge_weight", type=float, default=0.1)
    p.add_argument("--perceptual_weight", type=float, default=0.1)
    p.add_argument("--chroma_weight", type=float, default=1.0)
    p.add_argument("--no_pretrained", action="store_true", help="encoder from scratch (no ImageNet download)")
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_val", type=int, default=500, help="evenly spaced test pairs per set scored each epoch")
    p.add_argument("--steps_per_epoch", type=int, default=0, help="0 = full pass")
    add_common_args(p, CHECKPOINTS / "colorizer")
    args = p.parse_args(argv)
    if args.smoke:
        args.epochs, args.max_train, args.max_val, args.batch, args.size = 2, 8, 4, 2, 64
        args.steps_per_epoch, args.perceptual_weight, args.no_pretrained = 3, 0.0, True
    device, amp, out = setup(args)
    random.seed(args.seed)

    if args.data:
        sources = parse_sources(args.data)
    else:
        d = find_dir(load_config()["datasets"]["llvip"])
        if d is None:
            raise SystemExit("LLVIP not found: pass --data or set datasets.llvip in configs/infravision.yaml")
        sources = [("llvip", str(d), 1.0)]
    train_pairs, weights, vls = [], [], {}
    for name, path, w in sources:
        tr, va = llvip_pairs(path, "train"), llvip_pairs(path, "test")
        if not tr:
            raise SystemExit(f"{name}: no train pairs in {path}")
        random.Random(args.seed).shuffle(tr)
        if args.max_train:
            tr = tr[: args.max_train]
        va = va[:: max(1, len(va) // args.max_val)][: args.max_val]
        train_pairs += tr
        weights += [w / len(tr)] * len(tr)          # each set gets `w` share of the samples
        if va:
            vls[name] = DataLoader(LLVIPColor(va, args.size, False), args.batch, shuffle=False,
                                   num_workers=args.workers)
        print(f"{name}: {path}: train pairs {len(tr)} (sampling weight {w}), val pairs {len(va)}")
    primary = sources[0][0]
    if primary not in vls:
        raise SystemExit(f"primary set {primary} has no test pairs")
    n_draw = args.steps_per_epoch * args.batch if args.steps_per_epoch else len(train_pairs)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), n_draw, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    tl = DataLoader(LLVIPColor(train_pairs, args.size, True, args.geo_aug, args.thermal_aug), args.batch,
                    sampler=sampler,
                    num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)

    if args.reverse and args.thermal_aug:
        raise SystemExit("--thermal_aug changes the thermal image, which is the target in --reverse mode")
    G = build_model(not args.no_pretrained and not args.init, args.reverse).to(device)
    if args.init:
        from ..colorizer import load_thermovision
        G.load_state_dict(load_thermovision(args.init, "cpu", verbose=True)[0].state_dict())
        print("G initialised from", args.init)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    D = optD = None
    if args.gan_weight > 0:
        D = PatchD().to(device)
        optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    perc = None
    if args.perceptual_weight > 0:
        try:
            perc = VGGPerceptual().to(device)
        except Exception as e:  # no internet for the VGG weights
            print(f"WARNING: VGG16 weights unavailable ({e.__class__.__name__}); perceptual loss disabled")

    meta = dict(model="ThermoVisionUNet", architecture="pretrained_unet", encoder="resnet34", input_size=args.size,
                trained_on="+".join(n for n, _, _ in sources), perceptual=perc is not None,
                gan_weight=args.gan_weight, geo_aug=args.geo_aug, thermal_aug=args.thermal_aug)
    if args.reverse:
        meta.update(model="ThermoVisionUNet-rgb2thermal", in_ch=3, out_ch=1, direction="rgb2thermal")
    keeper = BestKeeper(out, "rgb2thermal" if args.reverse else "thermovision", "val_psnr", meta)
    resumer = Resumer(out, keeper, args.resume, G=G, opt=opt, sched=sched, scaler=scaler, D=D, optD=optD)

    watch = LossWatch()
    for epoch in range(resumer.start_epoch(), args.epochs + 1):
        G.train()
        tot = n = 0
        for step, (t, v) in enumerate(tl, 1):
            t, v = t.to(device, non_blocking=True), v.to(device, non_blocking=True)
            x, y = (v, t) if args.reverse else (t, v)           # reverse: RGB in, thermal out
            with autocast(device, amp):
                fake = G(x)
            fake = fake.float()
            loss = F.l1_loss(fake, y) + args.edge_weight * F.l1_loss(edges(fake), edges(y))
            if not args.reverse:
                loss = loss + args.chroma_weight * F.l1_loss(chroma(fake), chroma(y))
            if perc is not None:
                with autocast(device, amp):
                    loss = loss + args.perceptual_weight * perc(rgb3(fake) * 2 - 1, rgb3(y) * 2 - 1)
            # D always judges a (thermal, RGB) pair; the generated half is the model's output
            pair = (lambda gen: D(gen, v)) if args.reverse else (lambda gen: D(t, gen))
            gan_on = D is not None and epoch >= args.gan_start
            if gan_on:   # fp32 discriminator: small, and LSGAN in fp16 is touchy
                g_adv = pair(fake)
                loss = loss + args.gan_weight * F.mse_loss(g_adv, torch.ones_like(g_adv))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if gan_on:
                d_real, d_fake = D(t, v), pair(fake.detach())
                d_loss = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                optD.step()
            watch(loss.item())
            tot, n = tot + loss.item(), n + 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(tl)} loss {tot / n:.4f}", flush=True)
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
        sched.step()

        G.eval()
        per = {}
        with torch.inference_mode():
            for name, vl in vls.items():
                ps, ss, sat, sat_t, con, con_t = [], [], [], [], [], []
                for i, (t, v) in enumerate(vl):
                    x, y = (v, t) if args.reverse else (t, v)
                    with autocast(device, amp):
                        pred = G(x.to(device))
                    pred = pred.float().cpu().clamp(0, 1)
                    ps.append(psnr(pred, y))
                    ss.append(ssim(pred, y))
                    sat.append(saturation(pred))
                    sat_t.append(saturation(y))
                    con.append(luma(pred).flatten(1).std(1))
                    con_t.append(luma(y).flatten(1).std(1))
                    if i == 0:
                        save_image(torch.cat([rgb3(x), rgb3(pred), rgb3(y)]),
                                   out / f"val_samples_epoch{epoch:03d}_{name}.png", nrow=t.shape[0])
                per[name] = dict(psnr=torch.stack(ps).mean().item(), ssim=torch.stack(ss).mean().item(),
                                 saturation=torch.stack(sat).mean().item(),
                                 target_saturation=torch.stack(sat_t).mean().item(),
                                 contrast_ratio=(torch.cat(con).mean() / torch.cat(con_t).mean().clamp_min(1e-6)).item())
        pm = per[primary]
        m = dict(train_loss=tot / max(n, 1), val_psnr=pm["psnr"], val_ssim=pm["ssim"],
                 val_saturation=pm["saturation"], target_saturation=pm["target_saturation"],
                 contrast_ratio=pm["contrast_ratio"], primary=primary,
                 per_set={k: {a: round(b, 4) for a, b in v.items()} for k, v in per.items()})
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
        if len(per) > 1:
            print("   per set: " + " | ".join(f"{k} {v['psnr']:.2f} dB sat {v['saturation']:.3f}/{v['target_saturation']:.3f}"
                                             for k, v in per.items()), flush=True)
        resumer.end_epoch(epoch, args.epochs)
    resumer.finish()
    print("Best:", keeper.best, "->", out / "thermovision_best.pth")


if __name__ == "__main__":
    main()
