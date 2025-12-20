"""Loss functions."""


def ΔEOK(y, ŷ, ab_weight=2.0, scale=50.0):
    """Loss aggregating 3D Euclidean distance in the oklab space.

    This is intended to be a workhorse or main loss that will converge well.

    Oklab theoretically has a just-noticeable difference (JND) of ~0.02. For a
    loss function, this means that if each pixel individually is just
    noticeably wrong, before any scaling or tonemapping, the loss function
    could be scaled up 50× to return 1. We do that.

    ab_weight gives more (or less) weight difference in chroma. At 2, we are
    implementing what’s sometimes called ΔEOK2 or deltaEOK2 by default. See:
    https://github.com/w3c/csswg-drafts/issues/6642#issuecomment-945714988
    """
    return (
        (y[:, 0] - ŷ[:, 0]).square()
        + (ab_weight * (y[:, 1] - ŷ[:, 1])).square()
        + (ab_weight * (y[:, 2] - ŷ[:, 2])).square()
    ).sqrt().mean() * scale


def ΔEOK_squared(y, ŷ, ab_weight=2.0, scale=50.0):
    """Like ΔEOK but squared.

    Scaled so that an image with 1 JND of different at each pixel would have
    loss = 1, the same as with plain ΔEOK.
    """
    return (
        (
            (
                (y[:, 0] - ŷ[:, 0]).square()
                + (ab_weight * (y[:, 1] - ŷ[:, 1])).square()
                + (ab_weight * (y[:, 2] - ŷ[:, 2])).square()
            ).sqrt()
            * scale
        )
        .square()
        .mean()
    )
