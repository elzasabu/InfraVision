import argparse
import itertools
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import FlirPairedDataset, UnpairedDataset
from models.cyclegan import PatchDiscriminator, ResnetGenerator

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--size", type=int, default=256)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--lambda_cycle", type=float, default=10.0)
p.add_argument("--paired_weight", type=float, default=10.0,
               help="L1 weight on aligned FLIR pairs (0 = pure CycleGAN)")
p.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
p.add_argument("--workers", type=int, default=2)
p.add_argument("--out", default="checkpoints/cyclegan")
p.add_argument("--resume", action="store_true")
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

# A = thermal (1ch), B = visible RGB (3ch)
G_AB = ResnetGenerator(1, 3).to(DEVICE)   # thermal -> RGB (the colorizer)
G_BA = ResnetGenerator(3, 1).to(DEVICE)   # RGB -> thermal
D_A = PatchDiscriminator(1).to(DEVICE)
D_B = PatchDiscriminator(3).to(DEVICE)

opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))
# linear decay over the second half of training
decay = lambda e: 1.0 if e < args.epochs // 2 else 1.0 - (e - args.epochs // 2) / max(1, args.epochs - args.epochs // 2)
sched = [torch.optim.lr_scheduler.LambdaLR(o, decay) for o in (opt_G, opt_D)]

mse, l1 = nn.MSELoss(), nn.L1Loss()

start = 0
ckpt_path = os.path.join(args.out, "latest.pth")
if args.resume and os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    for m, k in ((G_AB, "G_AB"), (G_BA, "G_BA"), (D_A, "D_A"), (D_B, "D_B")):
        m.load_state_dict(ck[k])
    opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
    start = ck["epoch"] + 1
    for s in sched:
        for _ in range(start):
            s.step()
    print(f"Resumed from epoch {start}")

unpaired = DataLoader(UnpairedDataset("train", args.size), batch_size=args.batch, shuffle=True,
                      num_workers=args.workers, drop_last=True)
paired = None
if args.paired_weight > 0:
    paired = DataLoader(FlirPairedDataset("train", args.size), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    paired_iter = iter(paired)

step = 0
for epoch in range(start, args.epochs):
    for a, b in unpaired:
        a, b = a.to(DEVICE), b.to(DEVICE)

        # ---- generators ----
        fake_b, fake_a = G_AB(a), G_BA(b)
        pred_b, pred_a = D_B(fake_b), D_A(fake_a)
        loss_gan = mse(pred_b, torch.ones_like(pred_b)) + mse(pred_a, torch.ones_like(pred_a))
        loss_cyc = l1(G_BA(fake_b), a) + l1(G_AB(fake_a), b)
        loss_G = loss_gan + args.lambda_cycle * loss_cyc

        loss_pair = torch.zeros((), device=DEVICE)
        if paired is not None:
            try:
                t, v = next(paired_iter)
            except StopIteration:
                paired_iter = iter(paired)
                t, v = next(paired_iter)
            t, v = t.to(DEVICE), v.to(DEVICE)
            loss_pair = l1(G_AB(t), v) + l1(G_BA(v), t)
            loss_G = loss_G + args.paired_weight * loss_pair

        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        # ---- discriminators ----
        def d_loss(D, real, fake):
            pr, pf = D(real), D(fake.detach())
            return 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))

        loss_D = d_loss(D_B, b, fake_b) + d_loss(D_A, a, fake_a)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        step += 1
        if step % 50 == 0:
            print(f"ep {epoch+1}/{args.epochs} step {step} G {loss_G.item():.3f} gan {loss_gan.item():.3f} "
                  f"cyc {loss_cyc.item():.3f} pair {loss_pair.item():.3f} D {loss_D.item():.3f}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    for s in sched:
        s.step()
    with torch.no_grad():
        save_image(torch.cat([a.repeat(1, 3, 1, 1), fake_b, b], 0) * 0.5 + 0.5,
                   os.path.join(args.out, "samples", f"epoch_{epoch+1:03d}.png"), nrow=a.size(0))
    torch.save({"epoch": epoch, "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()}, ckpt_path)
    torch.save(G_AB.state_dict(), os.path.join(args.out, "colorizer_G_AB.pth"))
    if args.max_steps and step >= args.max_steps:
        break

print("Done. Colorizer weights:", os.path.join(args.out, "colorizer_G_AB.pth"))
