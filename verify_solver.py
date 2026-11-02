"""求解器稳健性检验（P0）· 评估与对比图：按求解器拆 3 面板，每面板 ours vs no_apf。

回答审稿人「换个 RL 算法，APF 的 OOD 尺寸鲁棒性（ours 相对 no_apf 的优势）还成立吗」。
布局：3 个子面板（PPO / SAC / TD3），每面板只有 2 条线（ours 绿实线 vs no_apf 红虚线），
横向对比「无论哪个求解器，ours 都平、no_apf 都崩，gap 一致」。

数据源：
- PPO 的 ours / no_apf：radius_generalization.csv（10 seed，主实验已产出）。
- SAC / TD3 的 ours / no_apf：models_solver/{SAC,TD3}_{ours,no_apf}_seed*.zip（train_solver.py 产出）。

产出：
- solver_radius.csv（SAC/TD3 的半径扫描原始数据，含 group 维度，10 seed 均值 + std）
- Paper_Figure_SolverRobustness.png（1×3 面板：安全成功率 vs 半径）
- Paper_Figure_SolverLearningCurve.png（1×3 面板：安全成功率 vs 训练步数）
"""
import os
import csv
import gymnasium as gym
import panda_gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles
from solver_common import flatten_obs
from ci_utils import ci_halfwidth_from_pop_std, ci_halfwidth_from_samples

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SUCCESS_THRESHOLD = 0.03
COLLISION_PENALTY = 0.2
STEP_COST = 0.05
MOTION = "random_walk"
TRAIN_RADIUS = 0.03
RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
SEEDS = list(range(1, 21))   # models/caches that exist are used; SAC/TD3 now up to 20 seeds
N_EVAL = 50
MODEL_DIR = "models_solver"
SCENARIO = "dynamic"

SOLVERS = ["PPO", "SAC", "TD3"]           # PPO 从 CSV 读；SAC/TD3 从模型扫
GROUPS = ["ours", "no_apf"]
GROUPS_KW = {
    "ours": dict(),
    "no_apf": dict(safety_weight=0.0, height_penalty=0.0),
}
GROUP_COLOR = {"ours": "#2ca02c", "no_apf": "#d62728"}
GROUP_LS = {"ours": "-", "no_apf": "--"}
GROUP_LABEL = {"ours": "ours (APF shaping)", "no_apf": "no_apf (no APF)"}
SOLVER_LABEL = {"PPO": "PPO", "SAC": "SAC", "TD3": "TD3"}


def sample_goals():
    """与 verify_radius_generalize.py 完全一致：预采样 50 个跨障碍随机目标（train radius 0.03）。"""
    goals = []
    for s in range(N_EVAL):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(
            raw, obstacles=get_obstacles(True, MOTION, radius=TRAIN_RADIUS),
            success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
            collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST, hide_obs=True)
        obs, _ = e.reset(seed=10000 + s)
        goals.append(e._sample_cross_obstacle_goal())
        e.close()
    return goals


def run_model(model, kw, radius, goals):
    """单模型在给定半径下跑 N_EVAL 个固定目标（与 verify_radius_generalize.py 同口径）。"""
    n_succ = n_coll = n_top = n_safe = n_through = n_collide_fail = n_safe_fail = 0
    paths = []
    min_clears = []
    for s, goal in enumerate(goals):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(
            raw, obstacles=get_obstacles(True, MOTION, radius=radius),
            success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
            collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST, hide_obs=True, **kw)
        obs, _ = e.reset(seed=10000 + s)
        obs = e._force_goal(obs, goal)

        traj = [obs["observation"][:3].copy()]
        ep_coll = ep_top = False
        min_clear = 1e9
        for _ in range(100):
            ee = obs["observation"][:3]
            for oc in e.obstacles:
                dc = float(np.hypot(ee[0] - oc["x"], ee[1] - oc["y"]))
                if 0.0 <= ee[2] <= oc["height"]:
                    min_clear = min(min_clear, dc - oc["radius"])
                if dc < oc["radius"] and ee[2] > oc["height"]:
                    ep_top = True
            a, _ = model.predict(flatten_obs(obs), deterministic=True)
            obs, r, term, trunc, info = e.step(a)
            obs = dict(obs)
            obs["desired_goal"] = goal.copy()
            traj.append(obs["observation"][:3].copy())
            if info.get("is_collision", False):
                ep_coll = True
            if trunc or np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD:
                break
        e.close()

        P = np.array(traj)
        paths.append(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
        min_clears.append(min_clear)
        reached = np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD
        collided = bool(ep_coll)
        if reached:
            n_succ += 1
            if collided:
                n_through += 1
            else:
                n_safe += 1
        else:
            if collided:
                n_collide_fail += 1
            else:
                n_safe_fail += 1
        n_coll += int(collided)
        n_top += int(ep_top)
    N = len(goals)
    return dict(succ=n_succ / N, coll=n_coll / N, top=n_top / N,
                safe=n_safe / N, through=n_through / N,
                path=float(np.mean(paths)), clear=float(np.mean(min_clears)))


def load_model(solver, path):
    return (SAC if solver == "SAC" else TD3).load(path)


CACHE_DIR = "scan_cache"


def radius_scan(solvers):
    """扫描指定求解器的 ours 与 no_apf，逐模型增量缓存到 scan_cache/*.npz（崩溃不丢）。

    每个模型扫完立即保存 scan_cache/{solver}_{group}_seed{seed}.npz，下次跳过已有缓存。
    返回 data[(solver, group, radius)] -> 指标 dict(list per seed)。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    goals = sample_goals()
    data = {}
    for solver in solvers:
        for group in GROUPS:
            for seed in SEEDS:
                model_path = f"{MODEL_DIR}/{solver}_{group}_seed{seed}"
                if not os.path.exists(model_path + ".zip"):
                    continue
                cache_path = f"{CACHE_DIR}/{solver}_{group}_seed{seed}.npz"
                if os.path.exists(cache_path):
                    c = np.load(cache_path)
                    for radius in RADII:
                        key = (solver, group, radius)
                        data.setdefault(key, {"coll": [], "succ": [], "top": [], "path": [],
                                              "clear": [], "safe": [], "through": []})
                        rk = f"r{radius}"
                        for k in data[key]:
                            data[key][k].append(float(c[f"{rk}_{k}"]))
                    print(f"[skip-cached] {solver} {group} seed{seed}", flush=True)
                    continue
                model = load_model(solver, model_path)
                cache = {}
                for radius in RADII:
                    r = run_model(model, GROUPS_KW[group], radius, goals)
                    key = (solver, group, radius)
                    data.setdefault(key, {"coll": [], "succ": [], "top": [], "path": [], "clear": [],
                                          "safe": [], "through": []})
                    for k in r:
                        data[key][k].append(r[k])
                        cache[f"r{radius}_{k}"] = r[k]
                np.savez(cache_path, **cache)
                del model
                import gc
                gc.collect()
                print(f"[done] {solver} {group} seed{seed}", flush=True)
    return data


def save_csv(data):
    rows = []
    for solver in ["SAC", "TD3"]:
        for group in GROUPS:
            for radius in RADII:
                d = data.get((solver, group, radius))
                if not d:
                    continue
                agg = {k: (float(np.mean(d[k])), float(np.std(d[k])))
                       for k in ("coll", "succ", "top", "path", "clear", "safe", "through")}
                rows.append([f"{solver}_{group}", radius,
                             agg["coll"][0], agg["coll"][1],
                             agg["succ"][0], agg["succ"][1],
                             agg["safe"][0], agg["safe"][1],
                             agg["through"][0], agg["through"][1],
                             agg["top"][0], agg["top"][1],
                             agg["path"][0], agg["path"][1],
                             agg["clear"][0], agg["clear"][1]])
                print(f"{solver}_{group:<10} r={radius:.3f} 撞柱率={agg['coll'][0]*100:5.1f}%  "
                      f"安全成功={agg['safe'][0]*100:5.1f}%  路径={agg['path'][0]:.3f}m", flush=True)
    with open("solver_radius.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "radius",
                    "collision_mean", "collision_std",
                    "success_mean", "success_std",
                    "safe_success_mean", "safe_success_std",
                    "through_mean", "through_std",
                    "over_top_mean", "over_top_std",
                    "path_mean", "path_std",
                    "clearance_mean", "clearance_std"])
        w.writerows(rows)


def load_ppo_radius():
    """从 radius_generalization.csv 读 PPO 的 ours/no_apf（四指标 mean+std）。
    返回 {group: {radius: {metric: (mean, std)}}}。"""
    out = {g: {} for g in GROUPS}
    with open("radius_generalization.csv") as f:
        for r in csv.DictReader(f):
            g = r["group"]
            if g not in GROUPS:
                continue
            n = int(r["n_seeds"])
            # 第二项为 95% CI 半宽（t 分布），论文图统一口径
            out[g][float(r["radius"])] = {
                "coll": (float(r["collision_mean"]), ci_halfwidth_from_pop_std(float(r["collision_std"]), n)),
                "safe": (float(r["safe_success_mean"]), ci_halfwidth_from_pop_std(float(r["safe_success_std"]), n)),
                "path": (float(r["path_mean"]), ci_halfwidth_from_pop_std(float(r["path_std"]), n)),
                "clear": (float(r["clearance_mean"]), ci_halfwidth_from_pop_std(float(r["clearance_std"]), n)),
            }
    return out


def _group_curve(solver, group, metric, data, ppo):
    """取某 (solver, group) 在 RADII 上的 (xs, ys, es)。"""
    xs, ys, es = [], [], []
    for rad in RADII:
        if solver == "PPO":
            d = ppo[group].get(rad)
            if d is None:
                continue
            m, s = d[metric]
        else:
            d = data.get((solver, group, rad))
            if not d or not d[metric]:
                continue
            m = float(np.mean(d[metric]))
            s = ci_halfwidth_from_samples(d[metric])   # 95% CI 半宽
        xs.append(rad)
        ys.append(m)
        es.append(s)
    return xs, ys, es


def plot_radius(data):
    ppo = load_ppo_radius()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    fig.suptitle("Solver robustness: safe success rate vs obstacle radius "
                 "(hidden obs, train r=0.03, 95% CI)", fontsize=14)
    for ci, solver in enumerate(SOLVERS):
        ax = axes[ci]
        for group in GROUPS:
            xs, ys, es = _group_curve(solver, group, "safe", data, ppo)
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=GROUP_COLOR[group], ls=GROUP_LS[group],
                            lw=2.5, marker="o", ms=6, capsize=3, label=GROUP_LABEL[group])
        ax.axvline(TRAIN_RADIUS, color="gray", ls=":", lw=1.5, alpha=0.8)
        ax.set_xlabel("Obstacle radius (m)")
        if solver == "PPO":
            n_label = 30   # PPO main experiment
        else:
            ns = [len(data[(solver, g, TRAIN_RADIUS)]["safe"]) for g in GROUPS
                  if (solver, g, TRAIN_RADIUS) in data]
            n_label = min(ns) if ns else 0
        ax.set_title(f"({chr(97+ci)}) {SOLVER_LABEL[solver]} (n={n_label})", fontsize=13)
        ax.set_ylim(-0.03, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Safe success rate (reach ∧ no collision)")
    fig.tight_layout()
    out = "Paper_Figure_SolverRobustness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[已保存] {out}  +  solver_radius.csv")


def load_curve(solver, group):
    """返回 {timestep: {coll: [...], safe: [...]}}（各 seed 汇总）。"""
    by_ts = {}
    for seed in SEEDS:
        if solver == "PPO":
            path = f"curves_formal/{SCENARIO}_{group}_seed{seed}.csv"
        else:
            path = f"curves_solver/{solver}_{group}_seed{seed}.csv"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                ts = int(float(row["timestep"]))
                d = by_ts.setdefault(ts, {"coll": [], "safe": []})
                d["coll"].append(float(row["collision_rate"]))
                d["safe"].append(float(row.get("safe_success_rate", 1 - float(row["collision_rate"]))))
    return by_ts


def plot_learning_curve():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    fig.suptitle("Solver robustness: sample efficiency (safe success rate, hidden obs)",
                 fontsize=14)
    for ci, solver in enumerate(SOLVERS):
        ax = axes[ci]
        for group in GROUPS:
            by_ts = load_curve(solver, group)
            ts = sorted(by_ts)
            xs = np.array(ts) / 1000
            ys = np.array([np.mean(by_ts[t]["safe"]) for t in ts])
            es = np.array([np.std(by_ts[t]["safe"]) for t in ts])
            if len(xs):
                ax.errorbar(xs, ys, yerr=es, color=GROUP_COLOR[group], ls=GROUP_LS[group],
                            lw=2.5, marker="o", ms=5, capsize=3, label=GROUP_LABEL[group])
        ax.set_xlabel("Training steps (k)")
        ax.set_title(f"({chr(97+ci)}) {SOLVER_LABEL[solver]}", fontsize=13)
        ax.set_ylim(-0.03, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Safe success rate (reach ∧ no collision)")
    fig.tight_layout()
    out = "Paper_Figure_SolverLearningCurve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[已保存] {out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--solvers", nargs="+", default=["SAC", "TD3"])
    p.add_argument("--skip-scan", action="store_true", help="跳过扫描，直接从 scan_cache 聚合出图")
    args = p.parse_args()

    print("=" * 80)
    print(f"【求解器稳健性检验 · 半径扫描 + 3 面板对比图 · solvers={args.solvers}】")
    print("=" * 80)
    if args.skip_scan:
        # 只从缓存聚合（缓存已齐时用）
        data = {}
        for solver in args.solvers:
            for group in GROUPS:
                for seed in SEEDS:
                    cache_path = f"{CACHE_DIR}/{solver}_{group}_seed{seed}.npz"
                    if not os.path.exists(cache_path):
                        continue
                    c = np.load(cache_path)
                    for radius in RADII:
                        key = (solver, group, radius)
                        data.setdefault(key, {"coll": [], "succ": [], "top": [], "path": [],
                                              "clear": [], "safe": [], "through": []})
                        rk = f"r{radius}"
                        for k in data[key]:
                            data[key][k].append(float(c[f"{rk}_{k}"]))
    else:
        data = radius_scan(args.solvers)
    if data:
        save_csv(data)
        plot_radius(data)
    plot_learning_curve()


if __name__ == "__main__":
    main()
