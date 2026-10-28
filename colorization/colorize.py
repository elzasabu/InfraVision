import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

from models.cyclegan import ResnetGenerator

p = argparse.ArgumentParser()
p.add_argument("input", help="thermal image path")
p.add_argument("--weights", default="checkpoints/cyclegan/colorizer_G_AB.pth")
p.add_argument("--out", default="colorized.png")
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = ResnetGenerator(1, 3).to(DEVICE)
G.load_state_dict(torch.load(args.weights, map_location=DEVICE))
G.eval()

img = Image.open(args.input).convert("L")
w, h = img.size
# generator is fully convolutional; round to a multiple of 4 for the two stride-2 stages
x = T.Compose([T.Resize((h // 4 * 4, w // 4 * 4)), T.ToTensor(), T.Normalize([0.5], [0.5])])(img)
with torch.no_grad():
    y = G(x.unsqueeze(0).to(DEVICE))
save_image(y * 0.5 + 0.5, args.out)
print("Saved", args.out)
