"""Summarize the inflated-collision-radius control (no_apf reward, PPO, hidden obs, train rho = 0.03).

Reference rows (no_apf at default and ours) come from radius_cache/*.npz (30 seeds).
Config naming: no_apf_m{inflation mm}_p{penalty}_seed{s}.
A seed "learned a margin" if its safe success at rho = 0.055 is >= 50 %.
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


configs = sorted({re.findall(r"no_apf_(m[\d.]+_p[\d.]+)_seed", f)[0]
                  for f in glob.glob("inflate_cache/no_apf_m*_seed*.npz")},
                 key=lambda s: (float(s.split("_")[0][1:]), float(s.split("_")[1][1:])))
groups = {"no_apf (30)": load("radius_cache/no_apf_seed*.npz")}
for c in configs:
    groups[c] = load(f"inflate_cache/no_apf_{c}_seed*.npz")
groups["ours (30)"] = load("radius_cache/ours_seed*.npz")

print(f"{'group':14s} {'n':>3s} {'clear@.03 mm':>14s} {'path m':>8s} " +
      " ".join(f"safe@{r:.3f}" for r in RADII) + "  coll@.055  margin-seeds")
for name, rows in groups.items():
    if not rows:
        continue
    cl = stat(rows, "r0.03_clear") * 1000
    pa = stat(rows, "r0.03_path")
    safe = [stat(rows, f"r{r}_safe") * 100 for r in RADII]
    coll = stat(rows, "r0.055_coll") * 100
    ok = int((safe[-1] >= 50).sum())
    print(f"{name:14s} {len(rows):3d} {cl.mean():6.1f}+-{ci(cl):4.1f} {pa.mean():8.3f} " +
          " ".join(f"{s.mean():10.1f}" for s in safe) + f"  {coll.mean():8.1f}  {ok}/{len(rows)}")

print("\nper-seed (clearance mm at 0.03, safe% at 0.055):")
for name, rows in groups.items():
    if name in ("no_apf (30)", "ours (30)") or not rows:
        continue
    print(f"{name:14s}", [(round(c, 1), round(s))
                          for c, s in zip(stat(rows, "r0.03_clear") * 1000,
                                          stat(rows, "r0.055_safe") * 100)])

ref = stat(groups["ours (30)"], "r0.055_safe")
for name, rows in groups.items():
    if name in ("no_apf (30)", "ours (30)") or len(rows) < 3:
        continue
    x = stat(rows, "r0.055_safe")
    print(f"{name} vs ours at 0.055: Welch p={stats.ttest_ind(x, ref, equal_var=False).pvalue:.3g}, "
          f"MWU p={stats.mannwhitneyu(x, ref).pvalue:.3g}")
