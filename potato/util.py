from einops import rearrange
import torch


def cheap_half(x):
    # Fast 2× downsample
    return (
        x[..., 0::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 0::2]
        + x[..., 1::2, 1::2]
    ) / 4.0


def pile(x, factor):
    return rearrange(
        x, "... c (h f0) (w f1) -> ... (c f0 f1) h w", f0=factor, f1=factor
    )


def tile(x, factor):
    return rearrange(
        x, "... (c f0 f1) h w -> ... c (h f0) (w f1)", f0=factor, f1=factor
    )


# def cheap_db1_dwt(x):
#     # expects tensor like ..., 1, H, W
#     # returns tensor like ...., 4, H, W
#     ah = x[..., 0::2] + x[..., 1::2]
#     vd = x[..., 0::2] - x[..., 1::2]

#     a = ah[..., 0::2, :] + ah[..., 1::2, :]
#     h = ah[..., 0::2, :] - ah[..., 1::2, :]

#     v = vd[..., 0::2, :] + vd[..., 1::2, :]
#     d = vd[..., 0::2, :] - vd[..., 1::2, :]

#     return torch.concat([a / 2, h, v, d], dim=-3) / 2

# def cheap_multilayer_db1_dwt(x):
#     return torch.concat(list(cheap_db1_dwt(x[:, n:n+1]) for n in range(x.shape[-3])), dim=-3)
