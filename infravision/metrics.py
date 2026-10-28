"""Image quality metrics on tensors in [0, 1], shape Bx C x H x W."""
import torch
import torch.nn.functional as F


def psnr(pred, target, eps=1e-10):
    mse = F.mse_loss(pred.float(), target.float(), reduction="none").flatten(1).mean(1)
    return (10 * torch.log10(1.0 / (mse + eps))).mean()


def _gauss(size=11, sigma=1.5, channels=1, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(channels, 1, size, size).contiguous()


def ssim(pred, target, size=11, sigma=1.5):
    """Mean SSIM (Wang et al. 2004, Gaussian window), averaged over channels."""
    pred, target = pred.float(), target.float()
    c = pred.shape[1]
    w = _gauss(size, sigma, c, pred.device)
    mu_x = F.conv2d(pred, w, groups=c)
    mu_y = F.conv2d(target, w, groups=c)
    sxx = F.conv2d(pred * pred, w, groups=c) - mu_x ** 2
    syy = F.conv2d(target * target, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(pred * target, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean()
