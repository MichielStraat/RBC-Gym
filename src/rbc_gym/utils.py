import matplotlib
from enum import IntEnum


def colormap(value, vmin=1, vmax=2, colormap="turbo"):
    cmap = matplotlib.colormaps[colormap]
    value = (value - vmin) / (vmax - vmin)
    return cmap(value, bytes=True)[:, :, :3]


class RBCField(IntEnum):
    T = 0
    UX = 1
    UY = 2
