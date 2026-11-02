"""Shared style for all result figures in the paper.

Figures are drawn at their final print size so that fonts are 6.5-9 pt in the PDF:
full-width figures are W = 5.4 in wide (the elsarticle preprint text width; included with width=\\linewidth) and
single-panel figures are W_HALF = 3.5 in wide (included with width=0.65\\linewidth).
Group names follow the paper text exactly (ours, no_apf, no_progress, vanilla, no_xy_apf).
Each figure is written to paper/figures/ as a vector PDF (fonts embedded, Type 42); that
folder is the one uploaded to Overleaf.
"""
import os

import matplotlib as mpl
from PIL import Image

W = 5.4
W_HALF = 3.5
TRAIN_RADIUS = 0.03
RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]

COLORS = {"ours": "#2ca02c", "no_apf": "#d62728", "no_progress": "#1f77b4",
          "vanilla": "#9467bd", "no_xy_apf": "#ff7f0e", "classical": "#1f77b4"}
LINESTYLES = {"ours": "-", "no_apf": "--", "no_progress": ":", "vanilla": "-.",
              "no_xy_apf": (0, (3, 1, 1, 1)), "classical": "-."}


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def panel_label(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold")


def mark_train_radius(ax, where="top", x=TRAIN_RADIUS, label=True):
    """Dotted vertical line at the training radius.

    The text is horizontal. In a multi-panel figure pass label=True for the first
    panel only, so the annotation appears once.
    """
    ax.axvline(x, color="gray", ls=":", lw=1.0, alpha=0.9)
    if not label:
        return
    y, va = {"bottom": (0.05, "bottom"), "top": (0.96, "top")}[where]
    ax.text(x + 0.001, y, r"training $\rho$", ha="left", va=va,
            fontsize=6.5, color="dimgray", transform=ax.get_xaxis_transform())


def legend_below(fig, ax, ncol, bottom):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=[0, bottom, 1, 1])


def save(fig, name):
    """Write the figure as a vector PDF into paper/figures/ (name may end in .png or .pdf)."""
    out_dir = os.path.join("paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(name))[0]
    path = os.path.join(out_dir, base + ".pdf")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"[saved] {path}")


def export(name):
    """Legacy helper: move an already saved PNG into paper/figures/ as a 2000-px-wide RGB file."""
    im = Image.open(name).convert("RGB")
    if im.width > 2000:
        im = im.resize((2000, round(im.height * 2000 / im.width)), Image.LANCZOS)
    out_dir = os.path.join("paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    im.save(os.path.join(out_dir, os.path.basename(name)))
    os.remove(name)
    print(f"[saved] {name} ({im.width}x{im.height}) -> paper/figures/")
