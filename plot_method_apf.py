"""Method figure: 3D coordinates of the analytic potential.

x, y : horizontal plane. The floor is coloured by the horizontal potential
       Phi (eq:apf), which is 1 on the pillar and falls to 0 across d_safe.
z    : height. The pillar occupies 0 <= z <= h. Above z = h the rear face of
       the box is coloured by the height penalty H (eq:height); Phi is off.

The radial and vertical profiles sit in a side gutter so they do not cover
the 3D axes.

Numerics follow multi_obstacle_env.py / method.tex:
    rho = 0.03 m, h = 0.21 m, d_safe = 0.08 m,
    Phi = ((d_safe - max(d_c, 0)) / d_safe)^2
          for 0 <= z <= h and d_c < d_safe,
    H = z_bar/0.04 + (z_bar/0.04)^2,  z_bar = max(0, z - h).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle as fs

RHO = 0.030
H_PILLAR = 0.210
D_SAFE = 0.080
H_SCALE = 0.040
R_OUT = RHO + D_SAFE

HALF = 0.150          # half-width of the coordinate box, metres
Z_TOP = 0.300

C_PHI = np.array([0xC4, 0x4E, 0x52]) / 255.0
C_H = np.array([0x1F, 0x77, 0xB4]) / 255.0
C_EDGE = "#7a3033"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
})


def height_of(z):
    over = np.clip(np.asarray(z, dtype=float) - H_PILLAR, 0.0, None)
    return over / H_SCALE + (over / H_SCALE) ** 2


def phi_rgba(clearance):
    """Map horizontal clearance (m) to an RGBA floor colour. Phi = 0 is cream."""
    depth = (D_SAFE - np.clip(clearance, 0.0, None)) / D_SAFE
    phi = np.where(clearance < D_SAFE, depth ** 2, 0.0)
    cream = np.array([0.973, 0.957, 0.941])
    rgb = cream * (1.0 - phi[..., None] * 0.92) + C_PHI * (phi[..., None] * 0.92)
    return np.concatenate([rgb, np.ones(phi.shape + (1,))], axis=-1)


def h_rgba(z, z_ref):
    """Blue above the pillar top, fading in from a visible onset at z = h.

    The onset is lifted so the ceiling reads as starting at z = h; panel (c)
    carries the exact linear-plus-quadratic profile.
    """
    z = np.asarray(z, dtype=float)
    active = z > H_PILLAR
    mix = np.zeros(z.shape)
    href = float(height_of(z_ref))
    mix[active] = 0.22 + 0.70 * (height_of(z[active]) / href)
    white = np.ones(3)
    rgb = white * (1.0 - mix[..., None]) + C_H * mix[..., None]
    return np.concatenate([rgb, np.ones(mix.shape + (1,))], axis=-1)


# ---------------------------------------------------------------------
fig = plt.figure(figsize=(6.3, 4.63))
ax = fig.add_axes([0.0, 0.0, 0.63, 1.0], projection="3d")
ax.computed_zorder = False
ax_b = fig.add_axes([0.70, 0.60, 0.27, 0.30])
ax_c = fig.add_axes([0.70, 0.08, 0.27, 0.34])

# The floor is split into tiles. One surface would be sorted as a single
# artist and painted over the pillar base.
n = 301
nb = 10
g = np.linspace(-HALF, HALF, n)
X, Y = np.meshgrid(g, g)
cuts = np.linspace(0, n - 1, nb + 1, dtype=int)
for i in range(nb):
    for j in range(nb):
        sl_i = slice(cuts[i], cuts[i + 1] + 1)
        sl_j = slice(cuts[j], cuts[j + 1] + 1)
        xs, ys = X[sl_i, sl_j], Y[sl_i, sl_j]
        cl = np.hypot(xs[:-1, :-1], ys[:-1, :-1]) - RHO
        ax.plot_surface(xs, ys, np.zeros_like(xs), facecolors=phi_rgba(cl),
                        shade=False, linewidth=0, antialiased=True, zorder=2)

# Rear wall, only above the pillar top: H grows with z and does not depend on x.
zw = np.linspace(H_PILLAR, Z_TOP, 241)
xw = np.linspace(-HALF, HALF, 2)
Xw, Zw = np.meshgrid(xw, zw)
ax.plot_surface(Xw, np.full_like(Xw, HALF), Zw,
                facecolors=h_rgba(Zw[:-1, :-1], Z_TOP),
                shade=False, linewidth=0, antialiased=True, zorder=1)

# ---------------- pillar ----------------
nu, nv = 64, 28
th = np.linspace(0, 2 * np.pi, nu)
zv = np.linspace(0.0, H_PILLAR, nv)
TH, ZV = np.meshgrid(th, zv)
# zorder must exceed the floor tiles (2); with computed_zorder=False the
# default of 1 paints the floor over the pillar body.
ax.plot_surface(RHO * np.cos(TH), RHO * np.sin(TH), ZV,
                color=tuple(C_PHI), shade=True, linewidth=0,
                antialiased=True, alpha=1, zorder=3)

# flat top cap, cartesian so it does not show polar banding
cg = np.linspace(-RHO, RHO, 70)
CX, CY = np.meshgrid(cg, cg)
inside = (CX[:-1, :-1] ** 2 + CY[:-1, :-1] ** 2) <= RHO ** 2
cap_rgba = np.zeros(inside.shape + (4,))
cap_rgba[inside, :3] = C_PHI * 0.92
cap_rgba[inside, 3] = 1.0
ax.plot_surface(CX, CY, np.full_like(CX, H_PILLAR), facecolors=cap_rgba,
                shade=False, linewidth=0, antialiased=True, zorder=4)

# outer edge of Phi: a ring on the floor and a ring at the pillar top
ring = np.linspace(0, 2 * np.pi, 300)
# The floor ring is split at the camera: the half behind the pillar must sit
# below the pillar (zorder 3), the half in front above it.
AZIM = -58.0
facing = (np.cos(ring) * np.cos(np.deg2rad(AZIM))
          + np.sin(ring) * np.sin(np.deg2rad(AZIM))) > 0
for mask, zo in ((facing, 6), (~facing, 2.5)):
    ax.plot(np.where(mask, R_OUT * np.cos(ring), np.nan),
            np.where(mask, R_OUT * np.sin(ring), np.nan),
            np.full_like(ring, 0.003),
            color="#1f77b4", lw=1.15, ls=(0, (4, 2.2)), zorder=zo)
ax.plot(R_OUT * np.cos(ring), R_OUT * np.sin(ring),
        np.full_like(ring, H_PILLAR),
        color="#1f77b4", lw=1.15, ls=(0, (4, 2.2)), zorder=6)

tag = dict(zorder=10, bbox=dict(boxstyle="round,pad=0.12",
                                facecolor="white", edgecolor="none", alpha=0.92))
ax.text(0.072, -0.040, 0.035, r"$\Phi$", color="#7a3033", fontsize=8, **tag)
ax.text(0.132, -0.055, 0.035, r"$d_{\mathrm{safe}}$", color="#1f77b4",
        fontsize=7.5, **tag)
ax.text(0.0, 0.142, 0.185, r"$z = h$", color="#14365c", fontsize=7.5, **tag)
ax.text(-0.09, 0.11, 0.270, r"$H$", color="#14365c", fontsize=8, **tag)

ax.set_xlim(-HALF, HALF)
ax.set_ylim(-HALF, HALF)
ax.set_zlim(0.0, Z_TOP)
ax.set_xlabel(r"$x$ (m)", labelpad=-1)
ax.set_ylabel(r"$y$ (m)", labelpad=-1)
ax.set_zlabel(r"$z$ (m)", labelpad=-2)
ax.set_xticks([-0.10, 0.0, 0.10])
ax.set_yticks([-0.10, 0.0, 0.10])
ax.set_zticks([0.0, H_PILLAR, Z_TOP])
ax.set_zticklabels(["0", r"$h$", "0.30"])
ax.tick_params(axis="both", pad=-1, length=2, width=0.5)
ax.tick_params(axis="z", pad=1)
ax.view_init(elev=30, azim=AZIM)
ax.set_box_aspect((1.0, 1.0, 0.92))
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=-6)
ax.grid(False)

for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor((1, 1, 1, 0.0))
    pane.set_edgecolor("#d0d0d0")
ax.xaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.7)
ax.yaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.7)
ax.zaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.7)
ax.xaxis._axinfo["grid"]["linewidth"] = 0.4
ax.yaxis._axinfo["grid"]["linewidth"] = 0.4
ax.zaxis._axinfo["grid"]["linewidth"] = 0.4

# ---------------- (b) radial profile of Phi ----------------
r = np.linspace(0.0, HALF, 600)
dc = r - RHO
depth = (D_SAFE - np.clip(dc, 0.0, None)) / D_SAFE
phi_r = np.where(dc < D_SAFE, depth ** 2, 0.0)
ax_b.plot(r, phi_r, color=tuple(C_PHI), lw=1.4)
ax_b.axvline(RHO, color="#d0d0d0", lw=0.6)
ax_b.axvline(R_OUT, color="#1f77b4", lw=0.6, ls=(0, (3, 1.6)))
ax_b.set_xlim(0, HALF)
ax_b.set_ylim(-0.05, 1.18)
ax_b.set_xticks([0, RHO, R_OUT])
ax_b.set_xticklabels(["0", r"$\rho$", r"$\rho+d_{\mathrm{safe}}$"])
ax_b.set_yticks([0, 1])
ax_b.set_xlabel(r"radial distance $r$ (m)")
ax_b.set_ylabel(r"$\Phi$")
fs.panel_label(ax_b, "b")
ax_b.tick_params(width=0.5, length=2.2)
for s in ax_b.spines.values():
    s.set_linewidth(0.6)

# ---------------- (c) vertical profile of H ----------------
z = np.linspace(0.0, Z_TOP, 400)
ax_c.plot(height_of(z), z, color=tuple(C_H), lw=1.4)
ax_c.axhline(H_PILLAR, color="#8eabc4", lw=0.6, ls=(0, (3, 1.6)))
ax_c.set_ylim(0.0, Z_TOP)
ax_c.set_xlim(-0.25, 8.0)
ax_c.set_yticks([0.0, H_PILLAR, Z_TOP])
ax_c.set_yticklabels(["0", r"$h$", "0.30"])
ax_c.set_xticks([0, 3, 6])
ax_c.set_xlabel(r"$H$")
ax_c.set_ylabel(r"$z$ (m)")
fs.panel_label(ax_c, "c")
ax_c.tick_params(width=0.5, length=2.2)
for s in ax_c.spines.values():
    s.set_linewidth(0.6)

fig.savefig("Paper_Figure_APFPotential.png", dpi=300, facecolor="white")
fs.export("Paper_Figure_APFPotential.png")
