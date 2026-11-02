"""Full vs hidden observation: collision rate vs obstacle radius (fig:fullobs).

Full observation: full_obs_radius.csv (n_seeds per group in CSV, solid).
Hidden observation: radius_generalization.csv (30 seeds, dashed, semi-transparent).
Error bars: 95% CI.  Output: Paper_Figure_fullobs_vs_hidden.png
"""
import csv

import matplotlib.pyplot as plt

import figstyle as fs
from ci_utils import ci_halfwidth_from_pop_std

GROUPS = ["no_apf", "ours"]


def load(path):
    out, n_by_group = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["group"] not in GROUPS:
                continue
            out[(r["group"], float(r["radius"]))] = (
                float(r["collision_mean"]), float(r["collision_std"]))
            n_by_group[r["group"]] = int(r["n_seeds"])
    return out, n_by_group


def main():
    fs.apply()
    full, n_full = load("full_obs_radius.csv")
    hidden, n_hidden = load("radius_generalization.csv")

    fig, ax = plt.subplots(figsize=(fs.W_HALF, 3.0))
    for g in GROUPS:
        ys = [full[(g, r)][0] for r in fs.RADII]
        es = [ci_halfwidth_from_pop_std(full[(g, r)][1], n_full[g]) for r in fs.RADII]
        ax.errorbar(fs.RADII, ys, yerr=es, color=fs.COLORS[g], ls="-", marker="o",
                    label=f"{g}, full observation")
        ys = [hidden[(g, r)][0] for r in fs.RADII]
        es = [ci_halfwidth_from_pop_std(hidden[(g, r)][1], n_hidden[g]) for r in fs.RADII]
        ax.errorbar(fs.RADII, ys, yerr=es, color=fs.COLORS[g], ls="--", lw=1.3, marker="s",
                    ms=3.5, alpha=0.6, label=f"{g}, hidden observation")
    fs.mark_train_radius(ax, "top")
    ax.set_xlabel("Obstacle radius (m)")
    ax.set_ylabel("Collision rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, bbox_to_anchor=(1.0, 0.13), fontsize=6.8)
    fig.tight_layout()
    fs.save(fig, "Paper_Figure_fullobs_vs_hidden.png")


if __name__ == "__main__":
    main()
