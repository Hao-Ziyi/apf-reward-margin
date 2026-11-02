"""Learning curves (fig:curve): collision rate and safe success rate vs training steps.

Aggregates curves_formal/*.csv (30 seeds for ours/no_apf/..., see SEEDS). Each checkpoint
is evaluated on 20 goals at the training radius (train_multi.py, N_EVAL = 20), so the
rates are not directly comparable to the 50-goal sweep tables.
Error bars: 95% CI over seeds.  Output: Paper_Figure_LearningCurve.png
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from ci_utils import ci_halfwidth_from_samples

CURVE_DIR = "curves_formal"
SCENARIO = "dynamic"
SEEDS = list(range(1, 31))   # missing seeds are skipped automatically
GROUPS = ["no_apf", "ours", "no_progress", "vanilla"]


def load_group(group):
    by_ts = {}
    for seed in SEEDS:
        path = f"{CURVE_DIR}/{SCENARIO}_{group}_seed{seed}.csv"
        try:
            f = open(path)
        except FileNotFoundError:
            continue
        with f:
            for row in csv.DictReader(f):
                ts = int(float(row["timestep"]))
                d = by_ts.setdefault(ts, {"coll": [], "safe": []})
                d["coll"].append(float(row["collision_rate"]))
                d["safe"].append(float(row.get("safe_success_rate",
                                               1 - float(row["collision_rate"]))))
    return by_ts


def main():
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.W, 2.6))
    panels = [("coll", "Collision rate"), ("safe", "Safe success rate")]
    for idx, (key, ylabel) in enumerate(panels):
        ax = axes[idx]
        for group in GROUPS:
            by_ts = load_group(group)
            ts = sorted(by_ts)
            xs = np.array(ts) / 1000
            ys = np.array([np.mean(by_ts[t][key]) for t in ts])
            es = np.array([ci_halfwidth_from_samples(by_ts[t][key]) for t in ts])
            ax.errorbar(xs, ys, yerr=es, color=fs.COLORS[group], ls=fs.LINESTYLES[group],
                        marker="o", label=group)
        ax.set_xlabel("Training steps (k)")
        ax.set_ylabel(ylabel)
        fs.panel_label(ax, chr(97 + idx))
        ax.set_ylim(-0.03, 1.08)

    fs.legend_below(fig, axes[0], ncol=4, bottom=0.12)
    fs.save(fig, "Paper_Figure_LearningCurve.png")


if __name__ == "__main__":
    main()
