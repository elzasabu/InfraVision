"""Train EDSR super-resolution on POP thermal crops; keep the best by val PSNR.

    python -m infravision.train.sr                      # full run (GPU recommended)
    python -m infravision.train.sr --smoke              # 2 tiny epochs on CPU
    python -m infravision.train.sr --scale 2 --epochs 30 --amp

Pairs are made on the fly from POP frames (clean crop vs blurred, downscaled,
re-upscaled, noisy, JPEG copy). Train = POP train flights, val = POP val
flights, so no flight is in both. Writes checkpoints/sr/edsr_best.pth and
checkpoints/sr/metrics.json, which the pipeline loads first.
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..data.pop import POP_ROOT, POPSRCrops, pop_splits
from ..enhance import build_edsr
from ..metrics import psnr, ssim
from .common import CHECKPOINTS, BestKeeper, Resumer, add_common_args, autocast, setup


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", default=str(POP_ROOT))
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--per_image", type=int, default=4, help="random crops per frame per epoch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ssim_weight", type=float, default=0.2, help="adds (1 - SSIM) to the L1 loss")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--max_train", type=int, default=0, help="cap on training frames (0 = all)")
    p.add_argument("--max_val", type=int, default=300, help="cap on val frames")
    add_common_args(p, CHECKPOINTS / "sr")
    args = p.parse_args(argv)
    if args.smoke:
        args.epochs, args.max_train, args.max_val, args.batch, args.per_image = 2, 8, 4, 4, 2
        args.crop, args.channels, args.blocks = 64, 16, 2
    device, amp, out = setup(args)

    splits = pop_splits(args.pop, existing_only=True)
    train_recs = splits["train"][:: max(1, len(splits["train"]) // args.max_train)] if args.max_train else splits["train"]
    train_recs = train_recs[: args.max_train or None]
    val_recs = splits["val"][:: max(1, len(splits["val"]) // args.max_val)][: args.max_val]
    print(f"train frames {len(train_recs)} ({len({r['flight'] for r in train_recs})} flights), "
          f"val frames {len(val_recs)} ({len({r['flight'] for r in val_recs})} flights)")

    train_ds = POPSRCrops(train_recs, args.crop, args.scale, args.per_image, train=True)
    val_ds = POPSRCrops(val_recs, args.crop * 2, args.scale, 1, train=False, seed=args.seed)
    tl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers)

    model = build_edsr(1, args.channels, args.blocks, global_residual=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    meta = dict(model="EDSR", in_channels=1, channels=args.channels, num_blocks=args.blocks,
                global_residual=True, scale=args.scale, trained_on="POP train flights")
    keeper = BestKeeper(out, "edsr", "val_psnr", meta)
    resumer = Resumer(out, keeper, args.resume, model=model, opt=opt, sched=sched, scaler=scaler)

    # Baseline: what bicubic alone gives on the same val crops.
    with torch.no_grad():
        b_psnr = torch.stack([psnr(x, y) for x, y in vl]).mean().item()
    print(f"val PSNR of bicubic input: {b_psnr:.2f} dB")

    for epoch in range(resumer.start_epoch(), args.epochs + 1):
        model.train()
        tot = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            with autocast(device, amp):
                pred = model(x)
                loss = F.l1_loss(pred, y)
                if args.ssim_weight:
                    loss = loss + args.ssim_weight * (1 - ssim(pred, y))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item(), n + 1
        sched.step()

        model.eval()
        ps, ss = [], []
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).float().cpu()
                ps.append(psnr(pred, y))
                ss.append(ssim(pred, y))
        m = dict(train_loss=tot / max(n, 1), val_psnr=torch.stack(ps).mean().item(),
                 val_ssim=torch.stack(ss).mean().item(), val_psnr_bicubic=b_psnr)
        best = keeper.update(epoch, model.state_dict(), m, vars(args))
        print(f"epoch {epoch}/{args.epochs} loss {m['train_loss']:.4f} val PSNR {m['val_psnr']:.2f} dB "
              f"SSIM {m['val_ssim']:.4f}" + ("  *best*" if best else ""), flush=True)
        resumer.end_epoch(epoch, args.epochs)
    resumer.finish()
    print("Best:", keeper.best, "->", out / "edsr_best.pth")


if __name__ == "__main__":
    main()
