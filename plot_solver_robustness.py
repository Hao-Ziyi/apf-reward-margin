"""Solver robustness (fig:solver): safe success rate vs obstacle radius for PPO, SAC, TD3.

PPO: radius_generalization.csv (30 seeds). SAC / TD3: solver_radius.csv (20 seeds each,
written by verify_solver.py). Error bars: 95% CI.  Output: Paper_Figure_SolverRobustness.png
"""
import csv

import matplotlib.pyplot as plt

import figstyle as fs
from ci_utils import ci_halfwidth_from_pop_std

GROUPS = ["ours", "no_apf"]
N_OFFPOLICY = 20


def load():
    out = {}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            if r["group"] in GROUPS:
                out[("PPO", r["group"], float(r["radius"]))] = (
                    float(r["safe_success_mean"]),
                    ci_halfwidth_from_pop_std(float(r["safe_success_std"]), int(r["n_seeds"])))
    with open("solver_radius.csv") as f:
        for r in csv.DictReader(f):
            solver, _, group = r["group"].partition("_")
            if group in GROUPS:
                out[(solver, group, float(r["radius"]))] = (
                    float(r["safe_success_mean"]),
                    ci_halfwidth_from_pop_std(float(r["safe_success_std"]), N_OFFPOLICY))
    return out


def main():
    fs.apply()
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(fs.W, 2.4), sharey=True)
    for idx, solver in enumerate(["PPO", "SAC", "TD3"]):
        ax = axes[idx]
        for g in GROUPS:
            ys = [data[(solver, g, r)][0] for r in fs.RADII]
            es = [data[(solver, g, r)][1] for r in fs.RADII]
            ax.errorbar(fs.RADII, ys, yerr=es, color=fs.COLORS[g], ls=fs.LINESTYLES[g],
                        marker="o", label=g)
        fs.mark_train_radius(ax, "bottom", label=(idx == 0))
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_xticks([0.03, 0.04, 0.05])
        ax.set_title(f"({chr(97 + idx)}) {solver}", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(-0.03, 1.08)
    axes[0].set_ylabel("Safe success rate")

    fs.legend_below(fig, axes[0], ncol=2, bottom=0.13)
    fs.save(fig, "Paper_Figure_SolverRobustness.png")


if __name__ == "__main__":
    main()
