"""Recompute every number quoted in paper/experiments.tex, discussion.tex, conclusion.tex,
introduction.tex and main.tex (abstract) from the raw per-seed data, so the text can be
checked against the data instead of against intermediate summary CSVs.

Raw sources
  radius_cache/{group}_seed{k}.npz   PPO per-seed radius sweep (50 goals per seed)
  scan_cache/{SOLVER}_{group}_seed{k}.npz   SAC / TD3 per-seed radius sweep
  curves_formal/dynamic_{group}_seed{k}.csv  PPO learning curves
  curves_fullobs/*.csv, full_obs_radius.csv  full observation
  classic_apf_radius.csv                      classical APF planner

Conventions: std = sample std (ddof=1); CI = t-distribution 95 %; p = two-sided Welch t-test
on per-seed safe success rate.  Run:  python verify_paper_numbers.py
"""
import csv
import glob
import math
import os

import numpy as np
from scipy import stats

RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
PPO_GROUPS = ["ours", "no_apf", "no_progress", "vanilla", "no_xy_apf"]


def load_cache(prefix, group, seeds=None, folder="radius_cache"):
    out = {}
    files = sorted(glob.glob(f"{folder}/{prefix}{group}_seed*.npz"),
                   key=lambda f: int(f.split("_seed")[-1].split(".")[0]))
    for f in files:
        k = int(f.split("_seed")[-1].split(".")[0])
        c = np.load(f, allow_pickle=True)
        out[k] = {r: {m: float(c[f"r{r}_{m}"]) for m in
                      ("coll", "succ", "top", "path", "clear", "safe", "through")}
                  for r in RADII}
    return out


def arr(d, r, m):
    return np.array([d[k][r][m] for k in sorted(d)])


def ci(x):
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / math.sqrt(n)


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def mwu(a, b):
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


ppo = {g: load_cache("", g) for g in PPO_GROUPS}
print({g: len(v) for g, v in ppo.items()})

# ---------------------------------------------------------------- Table main
hdr("Table main: safe success rate (%) mean over seeds")
for g in PPO_GROUPS:
    print(f"{g:<12}", " ".join(f"{100 * arr(ppo[g], r, 'safe').mean():6.1f}" for r in RADII))

hdr("Collision rate (%)")
for g in PPO_GROUPS:
    print(f"{g:<12}", " ".join(f"{100 * arr(ppo[g], r, 'coll').mean():6.1f}" for r in RADII))

hdr("Success rate P(S) (%)")
for g in PPO_GROUPS:
    print(f"{g:<12}", " ".join(f"{100 * arr(ppo[g], r, 'succ').mean():6.1f}" for r in RADII))

hdr("max over seeds&radii of [P(C) - P(S and C)] in percentage points (collision-timeout rate)")
for g in PPO_GROUPS:
    m = max((arr(ppo[g], r, "coll") - arr(ppo[g], r, "through")).mean() for r in RADII) * 100
    m_seed = max((arr(ppo[g], r, "coll") - arr(ppo[g], r, "through")).max() for r in RADII) * 100
    print(f"{g:<12} max of seed-mean = {m:.2f} pp ; max single-seed = {m_seed:.2f} pp")

hdr("Over-top rate")
print({g: max(arr(ppo[g], r, "top").max() for r in RADII) for g in PPO_GROUPS})

# ---------------------------------------------------------------- Table margin
hdr("Table margin at rho=0.03: clearance mm (mean +- sample std, +- pop std), center distance")
for g in ["ours", "no_apf", "no_progress", "vanilla", "no_xy_apf"]:
    x = arr(ppo[g], 0.03, "clear") * 1000
    print(f"{g:<12} {x.mean():6.2f} +- {np.std(x, ddof=1):5.2f} (pop {np.std(x):5.2f})  center {x.mean() + 30:6.2f}")
x1 = arr(ppo["ours"], 0.03, "clear").mean() * 1000
x2 = arr(ppo["no_apf"], 0.03, "clear").mean() * 1000
print("clearance ratio ours/no_apf", x1 / x2, " center ratio", (x1 + 30) / (x2 + 30))
hdr("slope of mean min clearance vs rho; mean center distance per radius")
for g in ["ours", "no_apf"]:
    cs = [arr(ppo[g], r, "clear").mean() * 1000 for r in RADII]
    print(g, "clearance", np.round(cs, 2), "center", np.round([c + r * 1000 for c, r in zip(cs, RADII)], 2),
          "slope", np.polyfit(RADII, cs, 1)[0] / 1000)
hdr("path length (m) at rho=0.03")
for g in PPO_GROUPS:
    print(g, round(arr(ppo[g], 0.03, "path").mean(), 4))
hdr("per-seed fraction of seeds with min clearance < 0 at each radius, no_apf")
for r in RADII:
    print(r, np.mean(arr(ppo["no_apf"], r, "clear") < 0))

# ---------------------------------------------------------------- Significance
hdr("Table significance: ours vs no_apf (CI, Welch p, Mann-Whitney p)")
for r in RADII:
    a, b = arr(ppo["ours"], r, "safe"), arr(ppo["no_apf"], r, "safe")
    print(f"{r:<6} no_apf {100 * b.mean():5.1f} [{100 * (b.mean() - ci(b)):5.1f},{100 * (b.mean() + ci(b)):5.1f}]  "
          f"ours {100 * a.mean():5.1f} [{100 * (a.mean() - ci(a)):5.1f},{100 * (a.mean() + ci(a)):5.1f}]  "
          f"diff {100 * (a.mean() - b.mean()):+5.1f}  welch p={welch(a, b):.3g}  MWU p={mwu(a, b):.3g}")

hdr("no_xy_apf vs no_apf ; ours vs no_xy_apf")
for r in RADII:
    n, a, o = arr(ppo["no_xy_apf"], r, "safe"), arr(ppo["no_apf"], r, "safe"), arr(ppo["ours"], r, "safe")
    print(f"{r:<6} noxy-vs-noapf diff {100 * (n.mean() - a.mean()):+5.1f} p={welch(n, a):.3g} | "
          f"ours-vs-noxy diff {100 * (o.mean() - n.mean()):+5.1f} p={welch(o, n):.3g}")

hdr("ours vs vanilla (rho=0.03) ; no_progress vs ours")
for r in RADII:
    o, v, npg = arr(ppo["ours"], r, "safe"), arr(ppo["vanilla"], r, "safe"), arr(ppo["no_progress"], r, "safe")
    print(f"{r:<6} ours-vs-vanilla diff {100 * (o.mean() - v.mean()):+5.1f} p={welch(o, v):.3g} | "
          f"noprog-vs-ours diff {100 * (npg.mean() - o.mean()):+5.1f} p={welch(npg, o):.3g}")

# ---------------------------------------------------------------- Learning curves
hdr("Sample efficiency (curves_formal): first checkpoint from which collision stays 0")


def conv_step(path):
    rows = list(csv.DictReader(open(path)))
    ts = [int(float(r["timestep"])) for r in rows]
    cr = [float(r["collision_rate"]) for r in rows]
    for i in range(len(cr)):
        if all(c == 0 for c in cr[i:]):
            return ts[i]
    return None


for g in PPO_GROUPS:
    steps = []
    final = []
    for k in range(1, 31):
        p = f"curves_formal/dynamic_{g}_seed{k}.csv"
        if not os.path.exists(p):
            continue
        steps.append(conv_step(p))
        final.append(float(list(csv.DictReader(open(p)))[-1]["collision_rate"]))
    conv = [s for s in steps if s is not None]
    print(f"{g:<12} n={len(steps)} converged={len(conv)} median(converged)={np.median(conv) if conv else None} "
          f"max={max(conv) if conv else None}  median incl. non-converged as inf="
          f"{np.median([s if s is not None else np.inf for s in steps])}  final mean collision={100 * np.mean(final):.2f}%  "
          f"#final>0={sum(f > 0 for f in final)}")
    print("   steps histogram:", {s: steps.count(s) for s in sorted(set(steps), key=lambda x: (x is None, x))})

# ---------------------------------------------------------------- SAC/TD3
hdr("SAC / TD3 (scan_cache)")
for solver in ["SAC", "TD3"]:
    d = {g: load_cache(f"{solver}_", g, folder="scan_cache") for g in ["ours", "no_apf"]}
    print(solver, {g: len(v) for g, v in d.items()})
    for r in RADII:
        a, b = arr(d["ours"], r, "safe"), arr(d["no_apf"], r, "safe")
        print(f"  {r:<6} ours {100 * a.mean():5.1f}  no_apf {100 * b.mean():5.1f}  diff {100 * (a.mean() - b.mean()):+6.1f}  "
              f"welch p={welch(a, b):.3g}  MWU p={mwu(a, b):.3g}")
    for g in d:
        print("  path", g, round(arr(d[g], 0.03, "path").mean(), 3), "clear mm", round(arr(d[g], 0.03, "clear").mean() * 1000, 1))

# ---------------------------------------------------------------- Full obs
hdr("Full observation (full_obs_radius.csv)")
for r in csv.DictReader(open("full_obs_radius.csv")):
    print(r["group"], r["radius"], "coll", round(float(r["collision_mean"]) * 100, 1), "succ", round(float(r["success_mean"]) * 100, 1))
for g in ["ours", "no_apf"]:
    allc = []
    import glob
    files = sorted(glob.glob(f"curves_fullobs/dynamic_{g}_seed*.csv"))
    print(g, "full-obs curve files:", len(files))
    for fpath in files:
        rows = list(csv.DictReader(open(fpath)))
        allc.append([float(r["collision_rate"]) for r in rows])
    allc = np.array(allc)
    print(g, "mean collision by checkpoint (%)", np.round(allc.mean(0) * 100, 1))
    first = [next((i for i in range(allc.shape[1]) if (allc[s, i:] == 0).all()), None) for s in range(allc.shape[0])]
    print("  first all-zero checkpoint index per seed:", first)
    print("  checkpoint at which the mean collision rate is 0:", [(i + 1) * 15000 for i in range(allc.shape[1]) if allc[:, i].mean() == 0][:1])

# ---------------------------------------------------------------- Classical APF
hdr("Classical APF (classic_apf_radius.csv)")
for r in csv.DictReader(open("classic_apf_radius.csv")):
    print(r["radius"], "safe", r["safe_success_mean"], "coll", r["collision_mean"], "stuck", r["stuck_mean"])

# ---------------------------------------------------------------- Geometry
hdr("Geometry: corridor and straight-line crossings")
c1, c2 = np.array([-0.07, 0.045]), np.array([-0.19, 0.015])
dctr = np.linalg.norm(c1 - c2)
print("center distance", dctr)
for rho in (0.03, 0.055):
    print("rho", rho, "corridor mm", (dctr - 2 * rho) * 1000)
print("corridor closes at rho =", dctr / 2)
print("nearest possible center distance under +-15mm x random walk:",
      math.hypot((-0.07 - 0.015) - (-0.19 + 0.015), 0.03), "-> closes at rho=",
      math.hypot((-0.07 - 0.015) - (-0.19 + 0.015), 0.03) / 2)
