"""Attribution control (fig:xy): ours vs no_apf vs no_xy_apf.

no_apf removes the horizontal potential and the height penalty; no_xy_apf removes only the
horizontal potential. Source: radius_generalization.csv (30 seeds).
Output: Paper_Figure_NoXY_APF.png (a: safe success rate, b: minimum clearance).
"""
import csv

import matplotlib.pyplot as plt

import figstyle as fs
from ci_utils import ci_halfwidth_from_pop_std

GROUPS = ["ours", "no_apf", "no_xy_apf"]


def load():
    data = {}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            g = r["group"]
            if g not in GROUPS:
                continue
            n = int(r["n_seeds"])
            data[(g, float(r["radius"]))] = {
                "safe": (float(r["safe_success_mean"]),
                         ci_halfwidth_from_pop_std(float(r["safe_success_std"]), n)),
                "clear": (float(r["clearance_mean"]),
                          ci_halfwidth_from_pop_std(float(r["clearance_std"]), n)),
            }
    return data


def main():
    fs.apply()
    data = load()
    fig, axes = plt.subplots(1, 2, figsize=(fs.W, 2.6))
    panels = [("safe", "Safe success rate", 1.0, "bottom"),
              ("clear", "Minimum clearance (mm)", 1000.0, "bottom")]
    for idx, (key, ylabel, scale, where) in enumerate(panels):
        ax = axes[idx]
        for g in GROUPS:
            rads = [r for r in fs.RADII if (g, r) in data]
            ys = [data[(g, r)][key][0] * scale for r in rads]
            es = [data[(g, r)][key][1] * scale for r in rads]
            ax.errorbar(rads, ys, yerr=es, color=fs.COLORS[g], ls=fs.LINESTYLES[g],
                        marker="o", label=g)
        fs.mark_train_radius(ax, "bottom", label=(idx == 0))
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_ylabel(ylabel)
        fs.panel_label(ax, chr(97 + idx))
        if key == "safe":
            ax.set_ylim(-0.03, 1.08)

    fs.legend_below(fig, axes[0], ncol=3, bottom=0.12)
    fs.save(fig, "Paper_Figure_NoXY_APF.png")

    print("\nno_xy_apf vs no_apf safe success difference (should be ~0):")
    for rad in fs.RADII:
        m1 = data[("no_xy_apf", rad)]["safe"][0]
        m2 = data[("no_apf", rad)]["safe"][0]
        print(f"  r={rad:.3f}: no_xy_apf={m1*100:5.1f}%  no_apf={m2*100:5.1f}%  diff={(m1-m2)*100:+5.1f}pp")


if __name__ == "__main__":
    main()
