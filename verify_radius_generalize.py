"""泛化评估：障碍半径变化（尺寸鲁棒性）——论文主图「撞柱率 vs 半径」。

实验设定（对应 HANDOFF 第五节）：
- 训练：2 柱、hide_obs=True、半径 0.03、随机游走（模型由 train_multi.py 产出到 models_formal/）。
- 评估：扫描半径 [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]，每半径跑 N_EVAL 个随机目标。
- 机制：隐观测下 no_apf 只能背下"离柱心 0.03"的经验间距 → 半径变大撞柱率陡升；
        ours 的 APF 势场相对真实半径自动缩放 → 撞柱率基本贴地。

产出：models_formal/dynamic_* 模型 + 本脚本 → radius_generalization.csv + PNG 主图。

图设计说明（3 面板）：
- (a) 撞柱率 vs 半径：核心区分度，no_apf 随半径陡升、ours 恒 0。
- (b) 成功率 vs 半径：对照，预期恒 ~100%（撞柱不终止、柱无物理体）。
- (c) 最小 clearance vs 半径：安全裕度量化，ours 恒 >0 且斜率 -1，no_apf 随半径转负。
- 隐观测下半径不进观测 → eval 轨迹与半径无关，撞柱率变化纯来自「训练学到的安全裕度」差异。
"""
import os
import csv
import glob
import gymnasium as gym
import panda_gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SUCCESS_THRESHOLD = 0.03
COLLISION_PENALTY = 0.2   # 与 train_multi.py 一致（撞柱不终止、每步扣分）
STEP_COST = 0.05          # 时间惩罚（与 train_multi.py 一致）
MOTION = "random_walk"
TRAIN_RADIUS = 0.03
RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
N_EVAL = 50               # 每半径每 seed 的随机目标数（压方差）
MODEL_DIR = "models_formal"
SCENARIO = "dynamic"      # 泛化实验只用动态（随机游走）场景
CACHE_DIR = "radius_cache"  # 逐模型缓存（键 = 模型 mtime），补种子重扫时不重扫未变模型
METRICS = ["coll", "succ", "top", "path", "clear", "safe", "through"]

# 与 train_multi.py 完全一致的 5 组消融（no_xy_apf 为归因控制，只进 CSV/单独图，不进主图）
GROUPS = [
    ("no_apf", dict(safety_weight=0.0, height_penalty=0.0)),
    ("ours", dict()),
    ("no_progress", dict(progress_weight=0.0)),
    ("vanilla", dict(smoothness_weight=0.0, safety_weight=0.0, height_penalty=0.0,
                     success_bonus=0.0, progress_weight=0.0)),
    ("no_xy_apf", dict(safety_weight=0.0)),
]

# 主图只画 4 组核心消融；no_xy_apf 作为独立归因 control 由 plot_no_xy_attribution.py 单独出图
MAIN_GROUPS = GROUPS[:4]

COLORS = {"no_apf": "#d62728", "ours": "#2ca02c",
          "no_progress": "#1f77b4", "vanilla": "#9467bd",
          "no_xy_apf": "#ff7f0e"}
LINESTYLES = {"no_apf": "--", "ours": "-", "no_progress": ":", "vanilla": "-.",
              "no_xy_apf": (0, (3, 1, 1, 1))}
LABELS = {
    "no_apf": "no_apf (no APF potential)",
    "ours": "ours (APF potential)",
    "no_progress": "no_progress (no progress reward)",
    "vanilla": "vanilla (no shaping)",
    "no_xy_apf": "no_xy_apf (no horizontal APF, keep height)",
}


def sample_goals():
    """用训练半径 0.03 预采样 N_EVAL 个跨障碍随机目标，所有半径共用同一批目标 → 公平对比。"""
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
    """单模型在给定半径下跑 N_EVAL 个固定目标。
    每集记录联合事件 (reached, collided)，精确统计四格：
      safe=到达∧无碰撞, through=到达∧撞柱, safe_fail=未达∧无碰撞, collide_fail=未达∧撞柱
    返回 dict: succ/coll/top/path/clear（旧边际）+ safe/through（新增联合）。"""
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
            a, _ = model.predict(obs, deterministic=True)
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


def group_seeds(name):
    """按可用模型文件自动检测该组的 seed 列表（P1 补种后 ours/no_apf=30，其余=10）。"""
    files = glob.glob(f"{MODEL_DIR}/{SCENARIO}_{name}_seed*.zip")
    return sorted(int(f.split("_seed")[-1].split(".zip")[0]) for f in files)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    goals = sample_goals()
    # data[(name, radius)] -> 每个指标一个 list（每 seed 一个值）
    data = {}
    n_seeds_by_group = {}
    for name, kw in GROUPS:
        seeds = group_seeds(name)
        n_seeds_by_group[name] = len(seeds)
        for seed in seeds:
            model_path = f"{MODEL_DIR}/{SCENARIO}_{name}_seed{seed}.zip"
            cache_path = f"{CACHE_DIR}/{name}_seed{seed}.npz"
            mtime = float(os.path.getmtime(model_path))
            cached = None
            if os.path.exists(cache_path):
                c = np.load(cache_path, allow_pickle=True)
                if float(c["model_mtime"]) == mtime:
                    cached = c
            if cached is not None:
                for radius in RADII:
                    key = (name, radius)
                    data.setdefault(key, {k: [] for k in METRICS})
                    for k in METRICS:
                        data[key][k].append(float(cached[f"r{radius}_{k}"]))
                print(f"[skip-cached] {name} seed{seed}", flush=True)
                continue
            model = PPO.load(model_path)
            cache = {"model_mtime": mtime}
            for radius in RADII:
                r = run_model(model, kw, radius, goals)
                key = (name, radius)
                data.setdefault(key, {k: [] for k in METRICS})
                for k in r:
                    data[key][k].append(r[k])
                    cache[f"r{radius}_{k}"] = r[k]
            np.savez(cache_path, **cache)
            del model
            print(f"[done] {name} seed{seed}", flush=True)

    if not data:
        print("没有可用的模型，先运行 train_multi.py 训练。")
        return

    # 聚合 + 保存 CSV（含 n_seeds 列，下游统计按组读实际 seed 数）
    rows = []
    for name, _ in GROUPS:
        for radius in RADII:
            d = data.get((name, radius))
            if not d:
                continue
            agg = {k: (float(np.mean(d[k])), float(np.std(d[k]))) for k in METRICS}
            rows.append([name, radius, n_seeds_by_group.get(name, len(d["coll"])),
                         agg["coll"][0], agg["coll"][1],
                         agg["succ"][0], agg["succ"][1],
                         agg["safe"][0], agg["safe"][1],
                         agg["through"][0], agg["through"][1],
                         agg["top"][0], agg["top"][1],
                         agg["path"][0], agg["path"][1],
                         agg["clear"][0], agg["clear"][1]])
            print(f"{name:<11} n={n_seeds_by_group[name]} r={radius:.3f} 撞柱率={agg['coll'][0]*100:5.1f}%  "
                  f"安全成功={agg['safe'][0]*100:5.1f}%  路径={agg['path'][0]:.3f}m", flush=True)

    with open("radius_generalization.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "radius", "n_seeds",
                    "collision_mean", "collision_std",
                    "success_mean", "success_std",
                    "safe_success_mean", "safe_success_std",
                    "through_mean", "through_std",
                    "over_top_mean", "over_top_std",
                    "path_mean", "path_std",
                    "clearance_mean", "clearance_std"])
        w.writerows(rows)

    # 绘图：2×2 四面板「撞柱率 / 安全成功率 / 路径长度 / 最小 clearance vs 半径」。
    # 安全成功率 = 到达∧无碰撞（诚实联合指标）；撞柱率是安全失败率；
    # 路径长度 = 裕度代价；clearance = 裕度机制。
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Size robustness under hidden obs (train r=0.03, dynamic random-walk)",
                 fontsize=14)

    panels = [
        ("coll", "Collision rate", "collision"),
        ("safe", "Safe success rate (reach ∧ no collision)", "safe success"),
        ("path", "Trajectory length (m)", "path length"),
        ("clear", "Min clearance (m)", "clearance"),
    ]
    for idx, (key, ylabel, _tag) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        for name, _ in MAIN_GROUPS:
            xs, ys, es, pts = [], [], [], []
            for radius in RADII:
                d = data.get((name, radius))
                if not d or not d[key]:
                    continue
                xs.append(radius)
                ys.append(float(np.mean(d[key])))
                es.append(float(np.std(d[key])))
                pts.append((radius, d[key]))
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=COLORS[name], ls=LINESTYLES[name],
                            lw=2.5, marker="o", ms=6, capsize=3, label=LABELS[name])
                for radius, vals in pts:
                    ax.scatter([radius] * len(vals), vals, color=COLORS[name], s=14,
                               alpha=0.22, edgecolors="none")
        ax.axvline(TRAIN_RADIUS, color="gray", ls=":", lw=1.5, alpha=0.8)
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+idx)}) {ylabel}", fontsize=12)
        ax.grid(True, alpha=0.3)
        if key in ("coll", "safe"):
            ax.set_ylim(-0.03, 1.05)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = "Paper_Figure8_RadiusGeneralization.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[已保存] {out}  +  radius_generalization.csv")


if __name__ == "__main__":
    main()
