"""Main size-robustness figure (fig:main): four panels vs obstacle radius, hidden observation.

(a) collision rate, (b) safe success rate, (c) trajectory length, (d) minimum clearance.
Error bars: 95% CI (t distribution) from radius_generalization.csv (*_std is the population
std, converted in ci_utils).  Output: Paper_Figure8_RadiusGeneralization.png
"""
import csv

import matplotlib.pyplot as plt

import figstyle as fs
from ci_utils import ci_halfwidth_from_pop_std

GROUPS = ["no_apf", "ours", "no_progress", "vanilla"]


def load():
    data, n_by_group = {}, {}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            g = r["group"]
            if g not in GROUPS:
                continue
            n = int(r.get("n_seeds", 10))
            n_by_group[g] = n
            data[(g, float(r["radius"]))] = {
                "coll": (float(r["collision_mean"]), float(r["collision_std"])),
                "safe": (float(r["safe_success_mean"]), float(r["safe_success_std"])),
                "path": (float(r["path_mean"]), float(r["path_std"])),
                "clear": (float(r["clearance_mean"]), float(r["clearance_std"])),
            }
    return data, n_by_group


def main():
    fs.apply()
    data, n_by_group = load()
    fig, axes = plt.subplots(2, 2, figsize=(fs.W, 5.0))

    # key, y label, scale, label position, row, col
    panels = [
        ("coll", "Collision rate", 1.0, "top", 0, 0),
        ("safe", "Safe success rate", 1.0, "bottom", 0, 1),
        ("path", "Trajectory length (m)", 1.0, "mid", 1, 0),
        ("clear", "Minimum clearance (mm)", 1000.0, "bottom", 1, 1),
    ]
    for key, ylabel, scale, where, ri, ci in panels:
        ax = axes[ri][ci]
        for g in GROUPS:
            ys = [data[(g, r)][key][0] * scale for r in fs.RADII]
            es = [ci_halfwidth_from_pop_std(data[(g, r)][key][1], n_by_group[g]) * scale
                  for r in fs.RADII]
            ax.errorbar(fs.RADII, ys, yerr=es, color=fs.COLORS[g], ls=fs.LINESTYLES[g],
                        marker="o", label=g)
        fs.mark_train_radius(ax, "top", label=(ri == 0 and ci == 0))
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_ylabel(ylabel)
        fs.panel_label(ax, chr(97 + ri * 2 + ci))
        if key in ("coll", "safe"):
            ax.set_ylim(-0.03, 1.08)

    fs.legend_below(fig, axes[0][0], ncol=4, bottom=0.06)
    fs.save(fig, "Paper_Figure8_RadiusGeneralization.png")


if __name__ == "__main__":
    main()
