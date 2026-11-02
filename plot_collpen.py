"""Collision-penalty control (fig:collpen): safe success rate and minimum clearance vs radius
for no_apf at penalties 0.2 / 5 / 10 and for ours (0.2).

Data: radius_cache/*.npz (ours, no_apf, 30 seeds) and collpen_cache/no_apf_p{5,10}_seed*.npz
(10 seeds). Output: Paper_Figure_CollPen.png
"""
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import figstyle as fs

SERIES = [
    ("ours, penalty 0.2", "radius_cache/ours_seed*.npz", fs.COLORS["ours"], "-"),
    ("no_apf, penalty 0.2", "radius_cache/no_apf_seed*.npz", fs.COLORS["no_apf"], "--"),
    ("no_apf, penalty 5", "collpen_cache/no_apf_p5_seed*.npz", "#ff7f0e", "-."),
    ("no_apf, penalty 10", "collpen_cache/no_apf_p10_seed*.npz", "#9467bd", ":"),
]


def load(pattern):
    files = sorted(glob.glob(pattern), key=lambda s: int(re.findall(r"seed(\d+)", s)[0]))
    return [np.load(f) for f in files]


def curve(rows, metric):
    return np.array([[float(r[f"r{rad}_{metric}"]) for rad in fs.RADII] for r in rows])


def ci_halfwidth(x, axis=0):
    n = x.shape[axis]
    return stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1, axis=axis) / np.sqrt(n)


def main():
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.W, 2.7))
    panels = [("safe", axes[0], "Safe success rate", "a", "bottom"),
              ("clear", axes[1], "Minimum clearance (mm)", "b", "bottom")]
    for metric, ax, ylabel, tag, where in panels:
        for label, pat, color, ls in SERIES:
            rows = load(pat)
            m = curve(rows, metric)
            if metric == "clear":
                m = m * 1000  # mm
            ax.errorbar(fs.RADII, m.mean(0), yerr=ci_halfwidth(m), color=color, ls=ls,
                        marker="o", label=f"{label} (n={m.shape[0]})", zorder=3)
            ax.scatter(np.tile(fs.RADII, m.shape[0]), m.ravel(), color=color, s=6,
                       alpha=0.25, edgecolors="none", zorder=2)
        fs.mark_train_radius(ax, "bottom", label=(tag == "a"))
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_ylabel(ylabel)
        fs.panel_label(ax, tag)
    axes[0].set_ylim(-0.03, 1.05)

    fs.legend_below(fig, axes[0], ncol=2, bottom=0.17)
    fs.save(fig, "Paper_Figure_CollPen.png")


if __name__ == "__main__":
    main()
