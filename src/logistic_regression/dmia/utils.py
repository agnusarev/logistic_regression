from typing import Any, List

import numpy as np
import pylab as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


def plot_surface(X: List[Any], y: List[Any], clf: Any) -> None:
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # type: ignore
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # type: ignore
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)  # type: ignore
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
