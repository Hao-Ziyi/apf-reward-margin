"""论文实验设置示意图（两张，训练设定 与 半径扫描 分开）。

图 1  Paper_Figure_EnvSetup.png      —— 训练设定（ρ=0.03 m）：
    (a) 俯视 xy：两柱（含随机游走范围）、起点、目标采样区域、示例目标、直线路径、
        以及两条「真实策略 rollout」轨迹（PPO seed1，确定性、隐观测；非手绘）。
    (b) 侧视 xz：柱高 0.21、目标高度带 0.08~0.13、翻顶通道 z>0.21、同一批真实轨迹的 xz 投影。
图 2  Paper_Figure_RadiusSweep.png   —— 评估半径扫描（ρ=0.03→0.055 m）：
    (a) 俯视：不同半径的柱轮廓叠画 + 两组策略在 ρ=0.03 学到的「最小到柱心距离」同心圆
        （ours 77.9 mm / no_apf 39.6 mm，取自 radius_generalization.csv 的 clearance_mean+0.03）。
    (b) 两柱间通道净宽 g(ρ)=d−2ρ 随半径变化，ρ≈0.0618 时通道闭合（解释扫描上限 0.055）。

所有几何与 multi_obstacle_env.py 一致；轨迹由 models_formal 真实模型 rollout 得到。

2026-10-18 版式修订（解决「缩进一栏后字太小 / 两个子图上下不齐」）：
  - 按最终尺寸出图：整图宽 6.3 英寸（≈ 论文单栏宽），字号 6.5–8 pt，缩放比 ≈ 1。
  - 用 fig.add_axes 手工指定两个子图的位置与高度，且数据范围按等比例反推 → 两个子图
    上沿、下沿严格对齐。
  - 随机游走范围的虚线加粗、不透明；图例统一放图下方，图内不再散落说明文字。
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from render_scene import rollout, FIXED_GOAL
import figstyle as fs

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

W_FIG = fs.W                                  # 整图宽（英寸）= 5.4，与 elsarticle preprint 正文宽一致
S = fs.W / 6.3                               # 相对旧 6.3 in 设计的缩放：只缩子图尺寸，字号不变

# ---- 几何常量（与 multi_obstacle_env.py 一致）----
A = np.array([-0.07, 0.045])
B = np.array([-0.19, 0.015])
R0 = 0.03
H = 0.21
AMP = 0.015                                   # 柱心沿 x 随机游走的幅度
START = np.array([0.0384, 0.0, 0.1974])       # env.reset 实测起点（obs[:3]）
GOAL = FIXED_GOAL.astype(float)               # 示例目标 (-0.33, 0.06, 0.10)
D_AB = float(np.linalg.norm(A - B))           # 两柱心距 ≈ 0.1237

C_PILLAR, C_PILLAR_FC = "#4d739e", "#d6e0ee"
C_GOAL = "#d9541a"
C_START = "#1a7a3a"
C_OURS, C_NOAPF = "#1a9e78", "#cc4040"
C_GRAY = "#6b6b6b"
WALK_LS = (0, (3, 1.6))                       # 随机游走范围的虚线样式（短划线，比点线醒目）


def seg_dist(P, Q, C):
    d = Q - P
    t = np.clip((C - P) @ d / (d @ d), 0, 1)
    return float(np.linalg.norm(P + t * d - C))


def get_rollouts():
    out = {}
    for name, path in [("no_apf", "models_formal/dynamic_no_apf_seed1.zip"),
                       ("ours", "models_formal/dynamic_ours_seed1.zip")]:
        out[name] = rollout(path)
        print(name, "rollout points", len(out[name]), "end", out[name][-1])
    return out


def pillar_patch(ax, c, r, **kw):
    ax.add_patch(Circle(c, r, **kw))


def add_axes_in(fig, h_fig, left, bottom, w, h):
    """按英寸指定子图位置（左下角 + 宽高）。"""
    return fig.add_axes([left / W_FIG, bottom / h_fig, w / W_FIG, h / h_fig])


# =====================================================================
# 图 1：训练设定
# =====================================================================
def fig_train(rolls=None, out="Paper_Figure_EnvSetup.png"):
    """rolls=None：只画环境几何（用于 Setup 小节）；传入 rolls：叠加真实策略轨迹（用于结果小节）。"""
    h_fig = 3.1
    pw = 2.62 * S                              # 子图宽（英寸）
    xspan, yspan = 0.55, 0.35                  # 两个子图数据跨度相同 → 等比例下框同高
    ph = pw * yspan / xspan                    # 子图高（英寸），由等比例反推
    bottom = 1.0
    fig = plt.figure(figsize=(W_FIG, h_fig))
    ax_a = add_axes_in(fig, h_fig, 0.48 * S, bottom, pw, ph)
    ax_b = add_axes_in(fig, h_fig, 0.48 * S + pw + 0.62 * S, bottom, pw, ph)

    # ---------- (a) 俯视 ----------
    ax = ax_a
    for c, name in [(A, "A"), (B, "B")]:
        for dx in (-AMP, AMP):                       # 随机游走范围（端点位置）
            pillar_patch(ax, (c[0] + dx, c[1]), R0, fill=False, ec=C_PILLAR, ls=WALK_LS,
                         lw=1.1, zorder=2)
        pillar_patch(ax, c, R0, fc=C_PILLAR_FC, ec=C_PILLAR, lw=1.4, zorder=3)
        ax.text(c[0], c[1], f"$\\phi_{name}$", ha="center", va="center", fontsize=8.5,
                color=C_PILLAR, fontweight="bold", zorder=4)
    ax.add_patch(Rectangle((-0.38, 0.03), 0.08, 0.12, fill=True, fc=C_GOAL, alpha=0.08,
                           ec=C_GOAL, ls="--", lw=0.9, zorder=1))
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color=C_GRAY, lw=1.2, zorder=2)
    for key, col in ([("no_apf", C_NOAPF), ("ours", C_OURS)] if rolls is not None else []):
        tr = rolls[key]
        ax.plot(tr[:, 0], tr[:, 1], "-o", color=col, lw=1.5, ms=2.6, zorder=5)
    ax.plot(*START[:2], "s", ms=6, color=C_START, zorder=6)
    ax.plot(*GOAL[:2], "*", ms=10, color=C_GOAL, zorder=6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fs.panel_label(ax, "a")
    ax.set_xlim(-0.43, 0.12)
    ax.set_ylim(-0.10, 0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
    ax.grid(True, alpha=0.3, lw=0.4)

    # ---------- (b) 侧视 ----------
    ax = ax_b
    for cx, name in [(A[0], "A"), (B[0], "B")]:
        for dx in (-AMP, AMP):                       # 随机游走范围（x 向）
            ax.add_patch(Rectangle((cx + dx - R0, 0), 2 * R0, H, fill=False, ec=C_PILLAR,
                                   ls=WALK_LS, lw=1.1, zorder=2))
        ax.add_patch(Rectangle((cx - R0, 0), 2 * R0, H, fc=C_PILLAR_FC, ec=C_PILLAR, lw=1.4, zorder=3))
        ax.text(cx, H - 0.014, f"$\\phi_{name}$", ha="center", va="top", fontsize=8.5,
                color=C_PILLAR, fontweight="bold", zorder=4)
    ax.add_patch(Rectangle((-0.43, 0.08), 0.55, 0.05, fc=C_GOAL, alpha=0.08, ec="none", zorder=1))
    ax.plot([START[0], GOAL[0]], [START[2], GOAL[2]], "--", color=C_GRAY, lw=1.2, zorder=2)
    for key, col in ([("no_apf", C_NOAPF), ("ours", C_OURS)] if rolls is not None else []):
        tr = rolls[key]
        ax.plot(tr[:, 0], tr[:, 2], "-o", color=col, lw=1.5, ms=2.6, zorder=5)
    ax.plot(START[0], START[2], "s", ms=6, color=C_START, zorder=6)
    ax.plot(GOAL[0], GOAL[2], "*", ms=10, color=C_GOAL, zorder=6)
    ax.axhline(H, color=C_GRAY, lw=0.7, ls=":")
    ax.text(-0.425, H + 0.007, "pillar top, $z$ = 0.21 m", fontsize=6.5, color=C_GRAY, va="bottom")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fs.panel_label(ax, "b")
    ax.set_xlim(-0.43, 0.12)
    ax.set_ylim(-0.04, 0.31)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
    ax.grid(True, alpha=0.3, lw=0.4)

    # ---------- 统一图例（图下方） ----------
    handles = [
        Line2D([], [], marker="s", ls="none", ms=5, color=C_START, label="start (EE), $z$ = 0.197 m"),
        Line2D([], [], marker="*", ls="none", ms=9, color=C_GOAL, label="example goal, $z$ = 0.10 m"),
        Patch(fc=C_GOAL, alpha=0.2, ec=C_GOAL, ls="--", lw=0.8,
              label="goal region (xy box; $z$ band 0.08–0.13 m)"),
        Line2D([], [], ls="--", color=C_GRAY, lw=1.2, label="direct line to example goal"),
    ]
    if rolls is not None:
        handles += [
            Line2D([], [], ls="-", marker="o", ms=2.6, color=C_NOAPF, lw=1.5, label="rollout: no_apf"),
            Line2D([], [], ls="-", marker="o", ms=2.6, color=C_OURS, lw=1.5, label="rollout: ours"),
        ]
    handles.append(Line2D([], [], ls=WALK_LS, color=C_PILLAR, lw=1.1,
                          label="pillar random-walk range ($\\pm$15 mm in $x$)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False, columnspacing=1.8, handlelength=2.0)

    fs.save(fig, out)
    print("[已保存]", out)


# =====================================================================
# 图 2：半径扫描
# =====================================================================
def read_center_dist():
    """r=0.03 行：最小到柱心距离 = clearance_mean + 0.03（单位 m）。"""
    d = {}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            if r["group"] in ("ours", "no_apf") and abs(float(r["radius"]) - 0.03) < 1e-9:
                d[r["group"]] = float(r["clearance_mean"]) + 0.03
    return d


def fig_sweep(with_margin=False, out="Paper_Figure_RadiusSweep.png"):
    """with_margin=False：只画评估半径与通道净宽（Setup 小节）；
    True：再叠加 ours / no_apf 学到的最小到柱心距离虚线圈（结果小节，可选）。"""
    radii = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
    cd = read_center_dist()
    print("center dist @0.03:", cd)
    cmap = plt.cm.Blues(np.linspace(0.35, 0.95, len(radii)))

    h_fig = 3.1
    pw_a, ph = 2.95 * S, 2.0 * S               # (a) 宽、两个子图共同高度（英寸）
    xspan_a = 0.36
    yspan_a = xspan_a * ph / pw_a              # 等比例反推 (a) 的 y 跨度 → 框高 = ph
    bottom = 0.98
    fig = plt.figure(figsize=(W_FIG, h_fig))
    ax_a = add_axes_in(fig, h_fig, 0.48 * S, bottom, pw_a, ph)
    ax_b = add_axes_in(fig, h_fig, 0.48 * S + pw_a + 0.6 * S, bottom, 2.2 * S, ph)

    # ---------- (a) 半径叠画 ----------
    ax = ax_a
    for r, col in reversed(list(zip(radii, cmap))):
        for c in (A, B):
            pillar_patch(ax, c, r, fc=col, alpha=0.30 if r > R0 else 0.55, ec=col, lw=1.0, zorder=2)
    for c, name in [(A, "A"), (B, "B")]:
        ax.text(c[0], c[1], f"$\\phi_{name}$", ha="center", va="center", fontsize=8.5,
                color="#102a43", fontweight="bold", zorder=5)
    ax.plot([A[0], B[0]], [A[1], B[1]], "-", color=C_GRAY, lw=0.8, zorder=3)
    ax.text((A[0] + B[0]) / 2, (A[1] + B[1]) / 2 - 0.012, f"$d$ = {D_AB*1000:.1f} mm",
            ha="center", fontsize=6.5, color=C_GRAY, zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for key, col in ([("ours", C_OURS), ("no_apf", C_NOAPF)] if with_margin else []):  # 学到的最小到柱心距离
        pillar_patch(ax, A, cd[key], fill=False, ec=col, ls="--", lw=1.5, zorder=6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fs.panel_label(ax, "a")
    ax.set_xlim(-0.30, 0.06)
    y_mid = 0.03
    ax.set_ylim(y_mid - yspan_a / 2, y_mid + yspan_a / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, lw=0.4)

    # ---------- (b) 通道净宽 ----------
    ax = ax_b
    xs = np.linspace(0.025, 0.066, 200)
    ax.plot(xs, (D_AB - 2 * xs) * 1000, "-", color=C_PILLAR, lw=1.4)
    ax.plot(radii, [(D_AB - 2 * r) * 1000 for r in radii], "o",
            color=C_PILLAR, ms=4, zorder=4, label="evaluation radii")
    ax.axhline(0, color="k", lw=0.6)
    r_close = D_AB / 2
    ax.axvline(r_close, color=C_NOAPF, ls="--", lw=1.0)
    ax.text(r_close - 0.0005, 38, f"corridor closes\nat $\\rho \\approx$ {r_close:.3f} m",
            color=C_NOAPF, fontsize=6.5, ha="right", linespacing=1.15)
    fs.mark_train_radius(ax, "bottom")
    for r in (0.03, 0.055):
        ax.annotate(f"{(D_AB-2*r)*1000:.1f} mm", xy=(r, (D_AB - 2 * r) * 1000),
                    xytext=(r + 0.0012, (D_AB - 2 * r) * 1000 + 5), fontsize=6.5, color=C_PILLAR)
    ax.set_xlabel("Obstacle radius (m)")
    ax.set_ylabel("Gap $d - 2\\rho$ (mm)")
    fs.panel_label(ax, "b")
    ax.set_xticks([0.03, 0.04, 0.05, 0.06])
    ax.set_xlim(0.025, 0.066)
    ax.set_ylim(-4, 72)
    ax.grid(True, alpha=0.3, lw=0.4)
    # ---------- 统一图例（图下方） ----------
    handles = [Line2D([], [], marker="o", ls="none", color=col, ms=5.5,
                      label="$\\rho$ = 30 mm (training)" if r == R0 else f"{r*1000:.0f} mm")
               for r, col in zip(radii, cmap)]
    if with_margin:
        # 左：评估半径（6 项）；右：两条虚线圈（各占一行）
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=2,
                   frameon=False, columnspacing=1.4, handletextpad=0.4, title="evaluation radius",
                   title_fontsize=6.5, alignment="left")
        margin_handles = [
            Line2D([], [], ls="--", color=C_OURS, lw=1.5,
                   label=f"ours: min. center distance {cd['ours']*1000:.1f} mm"),
            Line2D([], [], ls="--", color=C_NOAPF, lw=1.5,
                   label=f"no_apf: min. center distance {cd['no_apf']*1000:.1f} mm"),
        ]
        fig.legend(handles=margin_handles, loc="lower left", bbox_to_anchor=(0.50, 0.0), ncol=1,
                   frameon=False, handletextpad=0.5, title="dashed circles (around $\\phi_A$)",
                   title_fontsize=6.5, alignment="left")
    else:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
                   frameon=False, columnspacing=1.8, handletextpad=0.4, title="evaluation radius",
                   title_fontsize=6.5)

    fs.save(fig, out)
    print("[已保存]", out)


if __name__ == "__main__":
    # Setup 小节（只含几何，无训练结果）
    fig_train(None, "Paper_Figure_EnvSetup.png")
    fig_sweep(False, "Paper_Figure_RadiusSweep.png")
    # 结果小节（含训练后的真实策略轨迹 / 学到的最小到柱心距离）
    fig_train(get_rollouts(), "Paper_Figure_Rollouts.png")
    fig_sweep(True, "Paper_Figure_MarginCircles.png")
