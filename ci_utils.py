"""95% 置信区间工具（t 分布）。所有论文图的误差棒统一用它，口径与 plot_radius_ci.py 一致。

CSV 里的 *_std 是 np.std(ddof=0)（总体标准差），需先转样本标准差再算 CI。
"""
import math

import numpy as np
from scipy import stats


def ci_halfwidth_from_pop_std(std_pop, n):
    """由总体标准差(ddof=0) 和样本数 n 得 95% CI 半宽。"""
    if n < 2:
        return 0.0
    std_sample = std_pop * math.sqrt(n / (n - 1))
    return float(stats.t.ppf(0.975, df=n - 1) * std_sample / math.sqrt(n))


def ci_halfwidth_from_samples(values):
    """由原始样本得 95% CI 半宽（ddof=1）。"""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=n - 1) * np.std(v, ddof=1) / math.sqrt(n))
