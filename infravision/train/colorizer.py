"""Train the thermal -> RGB colorizer; keep the best by val PSNR.

    python -m infravision.train.colorizer                       # data/paired_disaster
    python -m infravision.train.colorizer --flir                # + FLIR aligned pairs
    python -m infravision.train.colorizer --smoke --device cpu  # 2 tiny epochs
    python -m infravision.train.colorizer --amp --adv_weight 0.01 --ssim_weight 0.5

Input: pre-enhanced thermal (1 channel, same CLAHE/denoise as the pipeline).
Output: 3-channel RGB. Loss = L1 + VGG16 perceptual (+ optional SSIM and
PatchGAN adversarial terms). Uses the ResnetGenerator from
colorization/models/cyclegan.py, so old CycleGAN weights still load.
Writes checkpoints/colorizer/colorizer_best.pth and metrics.json.

POP has no RGB, so the val metrics here are on held-out LLVIP pairs; POP is
judged by eye (docs/RESULTS.md).
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image

from ..config import load_config
from ..data.paired import DISASTER_DIR, PairedColor, disaster_pairs, flir_pairs
from ..enhance import build_colorizer, cyclegan_module, state_dict_of
from ..metrics import psnr, ssim
from .common import CHECKPOINTS, BestKeeper, Resumer, add_common_args, autocast, setup


class VGGPerceptual(nn.Module):
    """L1 between VGG16 features (relu1_2, relu2_2, relu3_3) of prediction and target."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for q in feats.parameters():
            q.requires_grad_(False)
        self.slices = nn.ModuleList([feats[:4], feats[4:9], feats[9:16]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        x = ((pred + 1) / 2 - self.mean) / self.std
        y = ((target + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DISASTER_DIR), help="built paired dataset (manifest.csv)")
    p.add_argument("--flir", action="store_true", help="also train on FLIR aligned pairs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--work_side", type=int, default=640, help="frames are resized to this longer side")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--perceptual_weight", type=float, default=0.1)
    p.add_argument("--ssim_weight", type=float, default=0.0)
    p.add_argument("--adv_weight", type=float, default=0.0, help="PatchGAN term (0 = off)")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--blocks", type=int, default=9)
    p.add_argument("--init", default="", help="start from these generator weights (e.g. old CycleGAN G_AB)")
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_val", type=int, default=200)
    p.add_argument("--steps_per_epoch", type=int, default=0, help="0 = full pass")
    add_common_args(p, CHECKPOINTS / "colorizer")
    args = p.parse_args(argv)
    if args.smoke:
        args.epochs, args.max_train, args.max_val, args.batch = 2, 8, 4, 2
        args.crop, args.work_side, args.ngf, args.blocks, args.steps_per_epoch = 64, 128, 16, 2, 3
    device, amp, out = setup(args)
    random.seed(args.seed)

    pre = load_config()["pre_enhance"]
    train_pairs = disaster_pairs("train", args.data)
    val_pairs = disaster_pairs("val", args.data)
    if args.flir:
        train_pairs += flir_pairs("train")
    random.Random(args.seed).shuffle(train_pairs)
    if args.max_train:
        train_pairs = train_pairs[: args.max_train]
    val_pairs = val_pairs[:: max(1, len(val_pairs) // args.max_val)][: args.max_val]
    print(f"train pairs {len(train_pairs)}, val pairs {len(val_pairs)}")

    train_ds = PairedColor(train_pairs, args.crop, args.work_side, True, pre)
    val_ds = PairedColor(val_pairs, args.crop, args.work_side, False, pre)
    tl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers)

    G = build_colorizer(args.ngf, args.blocks).to(device)
    if args.init:
        G.load_state_dict(state_dict_of(torch.load(args.init, map_location="cpu", weights_only=False))[0])
        print("initialised from", args.init)
    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    perc = None
    if args.perceptual_weight > 0:
        try:
            perc = VGGPerceptual().to(device)
        except Exception as e:  # no internet for the VGG weights
            print(f"WARNING: VGG16 weights unavailable ({e.__class__.__name__}); perceptual loss disabled")
    D = opt_D = None
    if args.adv_weight > 0:
        D = cyclegan_module().PatchDiscriminator(3).to(device)
        opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    meta = dict(model="ResnetGenerator", in_ch=1, out_ch=3, ngf=args.ngf, n_blocks=args.blocks,
                work_side=args.work_side, pre_enhance=pre, perceptual=perc is not None,
                trained_on="data/paired_disaster clean RGB" + (" + FLIR" if args.flir else ""))
    keeper = BestKeeper(out, "colorizer", "val_psnr", meta)
    resumer = Resumer(out, keeper, args.resume, G=G, opt=opt, sched=sched, scaler=scaler, D=D, opt_D=opt_D)

    for epoch in range(resumer.start_epoch(), args.epochs + 1):
        G.train()
        tot = n = 0
        for step, (t, v) in enumerate(tl, 1):
            t, v = t.to(device), v.to(device)
            with autocast(device, amp):
                fake = G(t)
                loss = F.l1_loss(fake, v)
                if perc is not None:
                    loss = loss + args.perceptual_weight * perc(fake, v)
                if args.ssim_weight:
                    loss = loss + args.ssim_weight * (1 - ssim((fake + 1) / 2, (v + 1) / 2))
                if D is not None:
                    pf = D(fake)
                    loss = loss + args.adv_weight * F.mse_loss(pf, torch.ones_like(pf))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if D is not None:
                with autocast(device, amp):
                    pr, pf = D(v), D(fake.detach())
                    ld = 0.5 * (F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
                opt_D.zero_grad(set_to_none=True)
                scaler.scale(ld).backward()
                scaler.step(opt_D)
                scaler.update()
            tot, n = tot + loss.item(), n + 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(tl)} loss {tot / n:.4f}", flush=True)
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
        sched.step()

        G.eval()
        ps, ss = [], []
        with torch.no_grad():
            for i, (t, v) in enumerate(vl):
                pred = ((G(t.to(device)).float().cpu().clamp(-1, 1)) + 1) / 2
                tgt = (v + 1) / 2
                ps.append(psnr(pred, tgt))
                ss.append(ssim(pred, tgt))
                if i == 0:
                    save_image(torch.cat([((t + 1) / 2).repeat(1, 3, 1, 1), pred, tgt]),
                               out / f"val_samples_epoch{epoch:03d}.png", nrow=t.shape[0])
        m = dict(train_loss=tot / max(n, 1), val_psnr=torch.stack(ps).mean().item(),
                 val_ssim=torch.stack(ss).mean().item())
        best = keeper.update(epoch, G.state_dict(), m, vars(args))
        print(f"epoch {epoch}/{args.epochs} loss {m['train_loss']:.4f} val PSNR {m['val_psnr']:.2f} dB "
              f"SSIM {m['val_ssim']:.4f}" + ("  *best*" if best else ""), flush=True)
        resumer.end_epoch(epoch, args.epochs)
    resumer.finish()
    print("Best:", keeper.best, "->", out / "colorizer_best.pth")


if __name__ == "__main__":
    main()
