from einops import rearrange


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
