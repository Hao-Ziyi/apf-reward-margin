"""Classical APF planner (sees the pillars) vs RL policies (hidden observation), fig:classic.

(a) safe success rate, (b) collision rate vs obstacle radius.
Sources: radius_generalization.csv (RL, 30 seeds) and classic_apf_radius.csv (deterministic).
Output: Paper_Figure_BaselineVsRL.png
"""
import csv

import matplotlib.pyplot as plt

import figstyle as fs
from ci_utils import ci_halfwidth_from_pop_std

PLANNER_LABEL = "classical APF (sees pillars)"


def main():
    fs.apply()
    with open("radius_generalization.csv") as f:
        rl_rows = [r for r in csv.DictReader(f) if r["group"] in ("ours", "no_apf")]
    with open("classic_apf_radius.csv") as f:
        cl_rows = list(csv.DictReader(f))

    def rl(group, radius, mean_key, std_key):
        row = [r for r in rl_rows if r["group"] == group and float(r["radius"]) == radius][0]
        return float(row[mean_key]), ci_halfwidth_from_pop_std(float(row[std_key]),
                                                               int(row["n_seeds"]))

    fig, axes = plt.subplots(1, 2, figsize=(fs.W, 2.7))
    panels = [("Safe success rate", "safe_success_mean", "safe_success_std", "bottom"),
              ("Collision rate", "collision_mean", "collision_std", "top")]
    for idx, (ylabel, mkey, skey, where) in enumerate(panels):
        ax = axes[idx]
        for g in ("ours", "no_apf"):
            vals = [rl(g, r, mkey, skey) for r in fs.RADII]
            ax.errorbar(fs.RADII, [v[0] for v in vals], yerr=[v[1] for v in vals],
                        color=fs.COLORS[g], ls=fs.LINESTYLES[g], marker="o", label=g)
        xc = [float(r["radius"]) for r in cl_rows]
        yc = [float(r[mkey]) for r in cl_rows]
        ax.plot(xc, yc, color=fs.COLORS["classical"], ls=fs.LINESTYLES["classical"],
                marker="s", label=PLANNER_LABEL)
        fs.mark_train_radius(ax, "bottom", label=(idx == 0))
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_ylabel(ylabel)
        fs.panel_label(ax, chr(97 + idx))
        ax.set_ylim(-0.03, 1.08)

    fs.legend_below(fig, axes[0], ncol=3, bottom=0.12)
    fs.save(fig, "Paper_Figure_BaselineVsRL.png")


if __name__ == "__main__":
    main()
