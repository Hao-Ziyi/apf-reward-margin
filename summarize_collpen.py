"""Summarize the collision-penalty control (no_apf reward, PPO, hidden obs, train rho = 0.03).

Reference rows (no_apf at P=0.2 and ours) come from radius_cache/*.npz (30 seeds).
A seed "learned a margin" if its safe success at rho = 0.055 is >= 50 % (ours: 89 %, no_apf: 9.5 %).
"""
import glob
import re
import numpy as np
from scipy import stats

RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern), key=lambda s: int(re.findall(r"seed(\d+)", s)[0])):
        c = np.load(f)
        rows.append({k: float(c[k]) for k in c.files if k.startswith("r")})
    return rows


def stat(rows, key):
    return np.array([r[key] for r in rows])


def ci(x):
    n = len(x)
    if n < 2:
        return float("nan")
    return stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)


groups = {
    "no_apf P=0.2 (30)": load("radius_cache/no_apf_seed*.npz"),
    "no_apf P=1 ": load("collpen_cache/no_apf_p1_seed*.npz"),
    "no_apf P=5 ": load("collpen_cache/no_apf_p5_seed*.npz"),
    "no_apf P=10": load("collpen_cache/no_apf_p10_seed*.npz"),
    "ours  P=0.2 (30)": load("radius_cache/ours_seed*.npz"),
}
print(f"{'group':18s} {'n':>3s} {'clear@.03 mm':>14s} {'path m':>8s} " +
      " ".join(f"safe@{r:.3f}" for r in RADII) + "  margin-seeds")
for name, rows in groups.items():
    if not rows:
        continue
    cl = stat(rows, "r0.03_clear") * 1000
    pa = stat(rows, "r0.03_path")
    safe = [stat(rows, f"r{r}_safe") * 100 for r in RADII]
    ok = int((safe[-1] >= 50).sum())
    print(f"{name:18s} {len(rows):3d} {cl.mean():6.1f}+-{ci(cl):4.1f} {pa.mean():8.3f} " +
          " ".join(f"{s.mean():10.1f}" for s in safe) + f"  {ok}/{len(rows)}")

print("\nper-seed (clearance mm at 0.03, safe% at 0.055):")
for name in ("no_apf P=5 ", "no_apf P=10"):
    rows = groups[name]
    print(name, [(round(c, 1), round(s))
                 for c, s in zip(stat(rows, "r0.03_clear") * 1000, stat(rows, "r0.055_safe") * 100)])

ref = stat(groups["ours  P=0.2 (30)"], "r0.055_safe")
for name in ("no_apf P=5 ", "no_apf P=10"):
    rows = groups[name]
    if len(rows) >= 2:
        x = stat(rows, "r0.055_safe")
        print(f"{name} vs ours at 0.055: Welch p={stats.ttest_ind(x, ref, equal_var=False).pvalue:.3g}, "
              f"MWU p={stats.mannwhitneyu(x, ref).pvalue:.3g}")
