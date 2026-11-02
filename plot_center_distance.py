"""Minimum distance to the pillar center vs obstacle radius (fig:margin).

d_center = clearance + radius. Under hidden observation the policy is open loop, so d_center
does not depend on the radius and each group is a horizontal line.
Source: radius_generalization.csv.  Output: Paper_Figure_CenterDistance.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from ci_utils import ci_halfwidth_from_pop_std

GROUPS = ["no_apf", "ours", "no_progress", "vanilla"]


def main():
    fs.apply()
    data = {}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            data[(r["group"], float(r["radius"]))] = (
                float(r["clearance_mean"]),
                ci_halfwidth_from_pop_std(float(r["clearance_std"]), int(r["n_seeds"])))

    fig, ax = plt.subplots(figsize=(fs.W_HALF, 3.0))
    for g in GROUPS:
        rads = [r for r in fs.RADII if (g, r) in data]
        ys = [(data[(g, r)][0] + r) * 1000 for r in rads]       # mm
        es = [data[(g, r)][1] * 1000 for r in rads]
        ax.errorbar(rads, ys, yerr=es, color=fs.COLORS[g], ls=fs.LINESTYLES[g],
                    marker="o", label=g)
    ax.plot(fs.RADII, [r * 1000 for r in fs.RADII], color="black", ls=":", lw=1.2,
            alpha=0.7, label="pillar surface")
    ax.set_xlabel("Obstacle radius (m)")
    ax.set_ylabel("Minimum distance to pillar center (mm)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45), frameon=True, framealpha=0.9)
    fig.tight_layout()
    fs.save(fig, "Paper_Figure_CenterDistance.png")

    print("\nmean center distance per group (m):")
    for g in GROUPS:
        vals = [data[(g, r)][0] + r for r in fs.RADII if (g, r) in data]
        print(f"  {g:<11} mean={np.mean(vals):.4f}  range=[{min(vals):.4f}, {max(vals):.4f}]")


if __name__ == "__main__":
    main()
