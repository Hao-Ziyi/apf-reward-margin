"""Summarize the (d_safe, lambda_s) sensitivity runs (ours reward, PPO, hidden obs, train rho = 0.03).

Reference: ours at (0.08, 1.5) from radius_cache/ours_seed1..10 (same seeds as the sensitivity runs).
Cells: sens_cache/ours_d{mm}_l{lam}_seed{s}.npz.  Prints a table and writes sens_summary.csv.
"""
import csv
import glob
import re
import numpy as np
from scipy import stats

RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]


def load(files):
    rows = []
    for f in files:
        c = np.load(f)
        rows.append({k: float(c[k]) for k in c.files if k.startswith("r")})
    return rows


def arr(rows, key):
    return np.array([r[key] for r in rows])


def ci(x):
    n = len(x)
    return float("nan") if n < 2 else stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)


cells = {}
for f in glob.glob("sens_cache/ours_d*_l*_seed*.npz"):
    d, l = re.findall(r"ours_d([\d.]+)_l([\d.]+)_seed", f)[0]
    cells.setdefault((float(d), float(l)), []).append(f)

ref_files = [f"radius_cache/ours_seed{s}.npz" for s in range(1, 11)]
groups = {(80.0, 1.5): load(ref_files)}
for k in sorted(cells):
    groups[k] = load(sorted(cells[k], key=lambda s: int(re.findall(r"seed(\d+)", s)[0])))
noapf = load([f"radius_cache/no_apf_seed{s}.npz" for s in range(1, 11)])

head = ["d_safe_mm", "lambda_s", "n", "clear_mm", "clear_ci", "path_m"] + \
       [f"safe@{r:.3f}" for r in RADII] + ["coll@0.055", "seeds_safe>=50%@0.055"]
out = [head]
print(f"{'d_safe':>7s} {'lam':>5s} {'n':>3s} {'clear mm':>12s} {'path':>6s} " +
      " ".join(f"{'s@'+format(r, '.3f'):>8s}" for r in RADII) + "   coll@.055  ok-seeds")
for (d, l), rows in sorted(groups.items()):
    cl = arr(rows, "r0.03_clear") * 1000
    pa = arr(rows, "r0.03_path")
    safe = [arr(rows, f"r{r}_safe") * 100 for r in RADII]
    coll = arr(rows, "r0.055_coll") * 100
    ok = int((safe[-1] >= 50).sum())
    tag = " (reference)" if (d, l) == (80.0, 1.5) else ""
    print(f"{d:7.0f} {l:5.2f} {len(rows):3d} {cl.mean():6.1f}+-{ci(cl):4.1f} {pa.mean():6.3f} " +
          " ".join(f"{s.mean():8.1f}" for s in safe) + f"   {coll.mean():8.1f}  {ok}/{len(rows)}{tag}")
    out.append([d, l, len(rows), round(cl.mean(), 2), round(ci(cl), 2), round(pa.mean(), 4)] +
               [round(s.mean(), 2) for s in safe] + [round(coll.mean(), 2), f"{ok}/{len(rows)}"])

cl0 = arr(noapf, "r0.03_clear") * 1000
s0 = arr(noapf, "r0.055_safe") * 100
print(f"\nno_apf seeds 1-10: clearance {cl0.mean():.1f} mm, safe@0.055 {s0.mean():.1f}%")

ref = arr(groups[(80.0, 1.5)], "r0.055_safe")
for k, rows in sorted(groups.items()):
    if k == (80.0, 1.5) or len(rows) < 3:
        continue
    x = arr(rows, "r0.055_safe")
    print(f"cell {k} vs reference at 0.055: Welch p={stats.ttest_ind(x, ref, equal_var=False).pvalue:.3g}; "
          f"vs no_apf: Welch p={stats.ttest_ind(x * 100, s0, equal_var=False).pvalue:.3g}")

with open("sens_summary.csv", "w", newline="") as fh:
    csv.writer(fh).writerows(out)
print("wrote sens_summary.csv")

