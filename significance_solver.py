"""求解器稳健性检验的统计显著性（P0 收尾）：对 SAC / TD3 的半径扫描「安全成功率」
做 95% CI + Welch t-test（ours vs no_apf），与 PPO 主实验 significance_test.py 同口径。

背景：verify_solver.py 产出的 solver_radius.csv 里 safe_success_std 是 np.std(ddof=0) 的
总体标准差；统计推断需先转样本标准差（×√(n/(n−1))）。seed 数从 scan_cache 里逐个 group
数 npz 文件得出（ours/no_apf 补种后均为 10）。

回答审稿人：「APF 优势跨求解器成立」是否统计显著（不是种子运气）。
"""
import csv
import glob
import math
import os

from scipy import stats

RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
SOLVERS = ["SAC", "TD3"]
GROUPS = ["ours", "no_apf"]
GROUP_CN = {"ours": "ours (APF)", "no_apf": "no_apf"}


def n_seeds(tag):
    """从 scan_cache 数某 (solver, group) 实际扫了多少个 seed（npz 文件数）。"""
    return len(glob.glob(f"scan_cache/{tag}_seed*.npz"))


def load():
    """读 solver_radius.csv → {(solver, group, radius): (safe_mean, safe_std_pop)}。"""
    data = {}
    with open("solver_radius.csv") as f:
        for r in csv.DictReader(f):
            tag = r["group"]          # e.g. "SAC_ours"
            for solver in SOLVERS:
                for group in GROUPS:
                    if tag == f"{solver}_{group}":
                        data[(solver, group, float(r["radius"]))] = (
                            float(r["safe_success_mean"]),
                            float(r["safe_success_std"]),
                        )
    return data


def ci95(mean, std_pop, n):
    std_sample = std_pop * math.sqrt(n / (n - 1))
    se = std_sample / math.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return (mean - t * se, mean + t * se)


def welch(m1, s1, n1, m2, s2, n2):
    s1s = s1 * math.sqrt(n1 / (n1 - 1))
    s2s = s2 * math.sqrt(n2 / (n2 - 1))
    t_stat, p = stats.ttest_ind_from_stats(m1, s1s, n1, m2, s2s, n2, equal_var=False)
    return t_stat, p


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    data = load()
    for solver in SOLVERS:
        print("\n" + "=" * 100)
        print(f"【{solver}】安全成功率 95% CI + ours vs no_apf Welch t-test")
        print("=" * 100)
        n_ours = n_seeds(f"{solver}_ours")
        n_noapf = n_seeds(f"{solver}_no_apf")
        print(f"seed 数：ours={n_ours}, no_apf={n_noapf}")

        print(f"\n① 95% 置信区间（t 分布）")
        print(f"{'半径':>7}  {'ours':>24}  {'no_apf':>24}")
        for rad in RADII:
            m1, s1 = data[(solver, "ours", rad)]
            m2, s2 = data[(solver, "no_apf", rad)]
            lo1, hi1 = ci95(m1, s1, n_ours)
            lo2, hi2 = ci95(m2, s2, n_noapf)
            print(f"{rad:>7}  {m1*100:>6.1f}% [{lo1*100:>5.1f},{hi1*100:>5.1f}]  "
                  f"{m2*100:>6.1f}% [{lo2*100:>5.1f},{hi2*100:>5.1f}]")

        print(f"\n② Welch t-test（ours vs no_apf）")
        print(f"{'半径':>7}  {'ours':>9}  {'no_apf':>9}  {'差(百分点)':>9}  {'p 值':>10}  结论")
        for rad in RADII:
            m1, s1 = data[(solver, "ours", rad)]
            m2, s2 = data[(solver, "no_apf", rad)]
            _, p = welch(m1, s1, n_ours, m2, s2, n_noapf)
            diff = (m1 - m2) * 100
            print(f"{rad:>7}  {m1*100:>8.1f}%  {m2*100:>8.1f}%  "
                  f"{diff:>+8.1f}  {p:>10.4g}  {stars(p)}")


if __name__ == "__main__":
    main()
