"""统计显著性检验：从现有 10-seed 聚合数据（radius_generalization.csv）计算
   95% 置信区间 + Welch t-test 显著性。

背景：CSV 里的 *_std 是 np.std(..., ddof=0)（总体标准差，除以 n）。
统计推断需要样本标准差（除以 n-1），故先做转换：
    std_sample = std_pop * sqrt(n / (n-1))

输出两类结果（都基于「安全成功率」这个核心指标）：
  1. 每组 × 每半径的 95% 置信区间（t 分布，df = n-1）
  2. 关键对比的 Welch t-test p 值（ours vs no_apf / vanilla；no_progress vs no_apf）

p 值 < 0.05 即「差异显著」；< 0.001 记为「***」（极显著）。
"""
import csv
import math
from scipy import stats

RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
GROUPS = ["no_apf", "ours", "no_progress", "vanilla", "no_xy_apf"]
GROUP_CN = {"no_apf": "no_apf", "ours": "ours", "no_progress": "no_progress",
            "vanilla": "vanilla", "no_xy_apf": "no_xy_apf"}


def load():
    """读 CSV，返回 {(group, radius): (safe_mean, safe_std_pop, n_seeds)}。"""
    data = {}
    n_by_group = {}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            g = r["group"]
            if g not in GROUPS:
                continue
            n = int(r.get("n_seeds", 10))
            n_by_group[g] = n
            data[(g, float(r["radius"]))] = (
                float(r["safe_success_mean"]),
                float(r["safe_success_std"]),
                n,
            )
    return data, n_by_group


def ci95(mean, std_pop, n):
    """95% 置信区间（t 分布）。std_pop 转样本标准差。"""
    std_sample = std_pop * math.sqrt(n / (n - 1))
    se = std_sample / math.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return (mean - t * se, mean + t * se)


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t-test，返回 p 值。s1/s2 是总体标准差（ddof=0），先转样本。"""
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
    data, n_by_group = load()

    # ---- 1. 置信区间表 ----
    print("=" * 100)
    print(f"① 安全成功率 95% 置信区间（t 分布，n={n_by_group}）")
    print("=" * 100)
    header = f"{'半径':>7}" + "".join(f"{g:>22}" for g in GROUPS)
    print(header)
    for rad in RADII:
        row = f"{rad:>7}"
        for g in GROUPS:
            m, s, n = data[(g, rad)]
            lo, hi = ci95(m, s, n)
            row += f"{m*100:>7.1f}% [{lo*100:>5.1f},{hi*100:>5.1f}]"
        print(row)

    # ---- 2. 显著性检验 ----
    print("\n" + "=" * 100)
    print("② Welch t-test 显著性（比较安全成功率，seed 数按组）")
    print("=" * 100)
    pairs = [("ours", "no_apf", "ours vs no_apf（核心对比：APF 有没有用）"),
             ("ours", "vanilla", "ours vs vanilla"),
             ("no_progress", "no_apf", "no_progress vs no_apf（归因：不是 PBRS）"),
             ("no_xy_apf", "no_apf", "no_xy_apf vs no_apf（归因：height 无关，应不显著）"),
             ("ours", "no_xy_apf", "ours vs no_xy_apf（归因：水平 APF 有用，应显著）")]
    for g1, g2, label in pairs:
        print(f"\n【{label}】")
        print(f"{'半径':>7}  {g1:>11}  {g2:>11}  {'差(百分点)':>9}  {'p 值':>10}  结论")
        for rad in RADII:
            m1, s1, n1 = data[(g1, rad)]
            m2, s2, n2 = data[(g2, rad)]
            _, p = welch(m1, s1, n1, m2, s2, n2)
            diff = (m1 - m2) * 100
            print(f"{rad:>7}  {m1*100:>9.1f}%  {m2*100:>9.1f}%  "
                  f"{diff:>+8.1f}  {p:>10.4g}  {stars(p)}")


if __name__ == "__main__":
    main()
