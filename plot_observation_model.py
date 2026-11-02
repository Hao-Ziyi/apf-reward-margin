"""Schematic of the observation model (hidden vs full observation).

Layout:
  top    (a)(b) top-view scenes
  middle thin colour bars: the observation vector the policy actually receives
  bottom shared colour legend (matplotlib, auto-placed)

Style follows figstyle.py (6.3 in wide, 7-9 pt).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# ---------------- geometry (matches multi_obstacle_env.py) ----------------
EE = np.array([0.038, 0.0])
GOAL = np.array([-0.33, 0.06])
BASE = np.array([-0.60, 0.0])
PILLAR_A = np.array([-0.07, 0.045])
PILLAR_B = np.array([-0.19, 0.015])
RHO = 0.03

# Schematic 3-link arm, routed south of the pillars so the top-view
# silhouette does not cross φ_A / φ_B.
J1 = np.array([-0.40, -0.035])
J2 = np.array([-0.16, -0.040])
ARM = np.vstack([BASE, J1, J2, EE])

# ---------------- palette ----------------
C_EE = "#1f77b4"
C_VEL = "#7fb4d9"
C_GOAL = "#2ca02c"
C_OBS1 = "#c44e52"
C_OBS2 = "#e87d1e"
C_GHOST = "#9a9a9a"
C_ARM = "#555555"
C_TABLE = "#f3f3f3"

import figstyle as fs
fs.apply()
plt.rcParams['axes.grid'] = False

W_FIG = fs.W
fig = plt.figure(figsize=(W_FIG, 3.7))
gs = GridSpec(
    2, 2, figure=fig,
    height_ratios=[3.2, 0.72],
    hspace=0.50, wspace=0.18,
    left=0.08, right=0.98, top=0.93, bottom=0.17,
)


def draw_scene(ax, observed, ylabel=False):
    ax.set_aspect("equal")
    ax.set_xlim(-0.72, 0.14)
    ax.set_ylim(-0.12, 0.24)

    # table / workspace
    ax.add_patch(Rectangle((-0.70, -0.08), 0.80, 0.28,
                           fc=C_TABLE, ec="#dddddd", lw=0.5, zorder=0))

    # schematic arm
    ax.plot(ARM[:, 0], ARM[:, 1], color=C_ARM, lw=2.0, solid_capstyle="round",
            zorder=2)
    ax.plot(ARM[:-1, 0], ARM[:-1, 1], "o", ms=4.0, color=C_ARM, zorder=3)
    ax.add_patch(Rectangle((BASE[0] - 0.028, BASE[1] - 0.018), 0.056, 0.036,
                           fc="#333333", ec="none", zorder=3))

    # dashed start-to-goal line
    ax.plot([EE[0], GOAL[0]], [EE[1], GOAL[1]],
            color="#bbbbbb", lw=0.8, ls=(0, (3, 1.6)), zorder=1)

    # pillars
    for name, c, col in [("A", PILLAR_A, C_OBS1), ("B", PILLAR_B, C_OBS2)]:
        if observed:
            ax.add_patch(Circle(c, RHO, fc=col, ec="none", alpha=0.55, zorder=4))
            ax.text(c[0], c[1] + RHO + 0.028, r"$\phi_{%s}$" % name,
                    ha="center", va="bottom", fontsize=7.5, color=col, zorder=6)
        else:
            ax.add_patch(Circle(c, RHO, fc="none", ec=C_GHOST, ls=(0, (3, 1.6)),
                                lw=1.2, zorder=4))
            ax.text(c[0], c[1] + RHO + 0.028, r"$\phi_{%s}$" % name,
                    ha="center", va="bottom", fontsize=7.5, color=C_GHOST, zorder=6)

    # EE + velocity arrow (visual of \dot p), pointing away from the pillars
    ax.plot(*EE, marker="o", ms=6.5, mfc=C_EE, mec="white", mew=0.7, zorder=6)
    ax.annotate("", xy=EE + np.array([0.055, 0.045]), xytext=EE,
                arrowprops=dict(arrowstyle="-|>", color=C_VEL, lw=1.1,
                                mutation_scale=8), zorder=6)
    ax.text(EE[0] + 0.022, EE[1] - 0.032, r"$\mathbf{p}$",
            color=C_EE, fontsize=8, zorder=7)
    ax.text(EE[0] + 0.070, EE[1] + 0.058, r"$\dot{\mathbf{p}}$",
            color=C_VEL, fontsize=8, zorder=7)

    # goal
    ax.plot(*GOAL, marker="*", ms=13, mfc=C_GOAL, mec="white", mew=0.5, zorder=6)
    ax.text(GOAL[0], GOAL[1] + 0.032, r"$\mathbf{g}$",
            color=C_GOAL, fontsize=8, ha="center", zorder=7)

    ax.set_xlabel("$x$ (m)")
    if ylabel:
        ax.set_ylabel("$y$ (m)")
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0])
    ax.set_yticks([-0.05, 0.05, 0.15])


def draw_obs_bar(ax, segments, eq_label):
    """Thin stacked colour bar. segments: (n_dims, color, math_label)."""
    total = sum(n for n, _, _ in segments)
    ax.set_xlim(0, total)
    ax.set_ylim(-0.55, 1.15)
    x = 0
    for n, col, lab in segments:
        ax.add_patch(Rectangle((x, 0), n, 1.0, fc=col, ec="white", lw=1.0, zorder=2))
        ax.text(x + n / 2, 0.50, lab, ha="center", va="center",
                fontsize=8, color="white", zorder=3)
        ax.text(x + n / 2, -0.28, str(n), ha="center", va="top",
                fontsize=7, color="#555555")
        x += n
    ax.add_patch(Rectangle((0, 0), total, 1.0, fc="none", ec="#333333",
                           lw=0.8, zorder=4))
    ax.set_title(eq_label, fontsize=8, pad=3)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---- (a) hidden ----
ax_a = fig.add_subplot(gs[0, 0])
draw_scene(ax_a, observed=False, ylabel=True)
fs.panel_label(ax_a, "a")
ax_a.text(0.97, 0.93, "pillars withheld", ha="right", va="top",
          transform=ax_a.transAxes, fontsize=7, color=C_GHOST, style="italic")

ax_ab = fig.add_subplot(gs[1, 0])
draw_obs_bar(
    ax_ab,
    [(3, C_EE, r"$\mathbf{p}$"),
     (3, C_VEL, r"$\dot{\mathbf{p}}$"),
     (3, C_GOAL, r"$\mathbf{g}$")],
    r"$\mathbf{o}\in\mathbb{R}^{9}$",
)

# ---- (b) full ----
ax_b = fig.add_subplot(gs[0, 1])
draw_scene(ax_b, observed=True, ylabel=False)
fs.panel_label(ax_b, "b")

ax_bb = fig.add_subplot(gs[1, 1])
draw_obs_bar(
    ax_bb,
    [(3, C_EE, r"$\mathbf{p}$"),
     (3, C_VEL, r"$\dot{\mathbf{p}}$"),
     (3, C_GOAL, r"$\mathbf{g}$"),
     (4, C_OBS1, r"$\mathbf{f}_{1}$"),
     (4, C_OBS2, r"$\mathbf{f}_{2}$")],
    r"$\mathbf{o}\in\mathbb{R}^{9+4N}$",
)

handles = [
    Patch(facecolor=C_EE, edgecolor="none", label=r"position $\mathbf{p}$"),
    Patch(facecolor=C_VEL, edgecolor="none", label=r"velocity $\dot{\mathbf{p}}$"),
    Patch(facecolor=C_GOAL, edgecolor="none", label=r"goal $\mathbf{g}$"),
    Patch(facecolor=C_OBS1, edgecolor="none", label=r"pillar A  $\mathbf{f}_{1}$"),
    Patch(facecolor=C_OBS2, edgecolor="none", label=r"pillar B  $\mathbf{f}_{2}$"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.53, 0.01), handlelength=1.1, columnspacing=1.3)

fs.save(fig, "Paper_Figure_ObservationModel.png")
