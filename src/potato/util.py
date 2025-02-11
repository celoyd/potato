from einops import rearrange, reduce
import torch


def cheap_half(x):
    return reduce(x, "... c (h1 h2) (w1 w2) -> ... c h1 w1", "mean", h2=2, w2=2)


def tile(x, factor=2):
    times = int(torch.round(torch.log2(torch.tensor(factor))))
    for _ in range(times):
        x = rearrange(x, "... (c two dos) h w -> ... c (h two) (w dos)", two=2, dos=2)
    return x


def pile(x, factor=2):
    times = int(torch.round(torch.log2(torch.tensor(factor))))
    for _ in range(times):
        x = rearrange(x, "... c (h two) (w dos) -> ... (c two dos) h w", two=2, dos=2)
    return x
