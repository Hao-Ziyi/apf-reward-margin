"""全观测(hide_obs=False) 正式验证：10 seed × 150k × 存 learning curve + 半径扫描。

目的：回答「切全观测后，150k 有限预算下 no_apf 的 OOD 撞柱率是否依然崩」。
      若崩 → APF 的「样本效率 × OOD 尺寸鲁棒」优势在全观测下依然成立，可放心切全观测。

产出：
  curves_fullobs/dynamic_{group}_seed{1..10}.csv  学习曲线（每 15k 评估，train radius 0.03）
  models_fullobs/dynamic_{group}_seed{1..10}.zip
  full_obs_radius.csv                             半径扫描（6 半径 × 10 seed）
  Paper_Figure_fullobs_vs_hidden.png              全观测 vs 隐观测 撞柱率叠图
"""
import os
import csv
import gc
import gymnasium as gym
import panda_gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TIMESTEPS = 150000
CHUNK = 15000
SUCCESS_THRESHOLD = 0.03
COLLISION_PENALTY = 0.2
STEP_COST = 0.05
MOTION = "random_walk"
TRAIN_RADIUS = 0.03
RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
def _parse_seeds(txt):
    out = []
    for part in txt.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


# FULLOBS_SEEDS="11-30" trains/scans only these seeds (no aggregation); unset = 1..10 + aggregate all cached seeds.
_ENV_SEEDS = os.environ.get("FULLOBS_SEEDS")
SEEDS = _parse_seeds(_ENV_SEEDS) if _ENV_SEEDS else list(range(1, 11))
ALL_SEEDS = list(range(1, 101))
N_EVAL = 50
SCENARIO = "dynamic"
MODEL_DIR = "models_fullobs"
CURVE_DIR = "curves_fullobs"

GROUPS = [
    ("no_apf", dict(safety_weight=0.0, height_penalty=0.0)),
    ("ours", dict()),
]
HIDE_OBS = False        # ★ 全观测


def make_env():
    env = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    env.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
    return env


def evaluate_model(model, kw, n_goal=N_EVAL):
    """train radius 下的块间评估（全观测）。返回 (成功率, 撞柱率)。"""
    succ = n_coll = 0
    for s in range(n_goal):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, MOTION, radius=TRAIN_RADIUS),
                                 success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
                                 collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST,
                                 hide_obs=HIDE_OBS, **kw)
        obs, _ = e.reset(seed=10000 + s)
        goal = e._sample_cross_obstacle_goal()
        obs = e._force_goal(obs, goal)
        ep_coll = False
        for _ in range(100):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = e.step(a)
            obs = dict(obs)
            obs["desired_goal"] = goal.copy()
            if info.get("is_collision", False):
                ep_coll = True
            if trunc or np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD:
                break
        e.close()
        if np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD:
            succ += 1
        n_coll += int(ep_coll)
    return succ / n_goal, n_coll / n_goal


def train_one(name, kw, seed):
    env = MultiObstacleWrapper(
        make_env(), obstacles=get_obstacles(True, MOTION, radius=TRAIN_RADIUS),
        success_threshold=SUCCESS_THRESHOLD, collision_penalty=COLLISION_PENALTY,
        step_cost=STEP_COST, hide_obs=HIDE_OBS, **kw)
    model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=0.01)
    curve = []
    for i in range(TIMESTEPS // CHUNK):
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(i == 0))
        sr, cr = evaluate_model(model, kw)
        curve.append(((i + 1) * CHUNK, sr, cr))
    model.save(f"{MODEL_DIR}/{SCENARIO}_{name}_seed{seed}")
    env.close()
    with open(f"{CURVE_DIR}/{SCENARIO}_{name}_seed{seed}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "success_rate", "collision_rate"])
        w.writerows(curve)
    return model


def sample_goals():
    goals = []
    for s in range(N_EVAL):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, MOTION, radius=TRAIN_RADIUS),
                                 success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
                                 collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST, hide_obs=True)
        obs, _ = e.reset(seed=10000 + s)
        goals.append(e._sample_cross_obstacle_goal())
        e.close()
    return goals


def run_model(model, kw, radius, goals):
    n_succ = n_coll = 0
    for s, goal in enumerate(goals):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, MOTION, radius=radius),
                                 success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
                                 collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST,
                                 hide_obs=HIDE_OBS, **kw)
        # Pillar random-walk seed = 10000 + goal index, identical to verify_radius_generalize.py.
        # (Earlier version used the constant seed 10000 for all 50 goals, i.e. one pillar
        #  realisation, which differs from the hidden-observation protocol.)
        obs, _ = e.reset(seed=10000 + s)
        obs = e._force_goal(obs, goal)
        ep_coll = False
        for _ in range(100):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = e.step(a)
            obs = dict(obs)
            obs["desired_goal"] = goal.copy()
            if info.get("is_collision", False):
                ep_coll = True
            if trunc or np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD:
                break
        e.close()
        n_succ += int(np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD)
        n_coll += int(ep_coll)
    return n_succ / N_EVAL, n_coll / N_EVAL


N_AGG = [0]
CACHE_DIR = "fullobs_cache"   # per-model cache of the radius sweep (key = model mtime)


def main():
    import sys
    only = sys.argv[1:]   # optional: restrict the scan to some groups, e.g.  python verify_fullobs.py ours
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    goals = sample_goals()
    data = {}

    for name, kw in GROUPS:
        if only and name not in only:
            continue
        for seed in SEEDS:
            path = f"{MODEL_DIR}/{SCENARIO}_{name}_seed{seed}"
            if not os.path.exists(path + ".zip"):
                print(f"[训练] full-obs {name} seed{seed} ...", flush=True)
                train_one(name, kw, seed)
            cache_path = f"{CACHE_DIR}/{name}_seed{seed}.npz"
            mtime = float(os.path.getmtime(path + ".zip"))
            if os.path.exists(cache_path) and float(np.load(cache_path)["model_mtime"]) == mtime:
                print(f"[skip-cached] full-obs {name} seed{seed}", flush=True)
                continue
            model = PPO.load(path)
            cache = {"model_mtime": mtime}
            for radius in RADII:
                s, c = run_model(model, kw, radius, goals)
                cache[f"r{radius}_succ"] = s
                cache[f"r{radius}_coll"] = c
            np.savez(cache_path, **cache)
            print(f"[done] full-obs {name} seed{seed}", flush=True)
            del model
            gc.collect()

    if _ENV_SEEDS:
        return   # partial run; aggregate later with FULLOBS_SEEDS unset
    # aggregate every seed that is cached for ALL groups
    agg_seeds = [sd for sd in ALL_SEEDS
                 if all(os.path.exists(f"{CACHE_DIR}/{g}_seed{sd}.npz") for g, _ in GROUPS)]
    print("aggregating seeds:", len(agg_seeds), flush=True)
    N_AGG[0] = len(agg_seeds)
    for name, _ in GROUPS:
        for seed in agg_seeds:
            cache_path = f"{CACHE_DIR}/{name}_seed{seed}.npz"
            c = np.load(cache_path)
            for radius in RADII:
                key = (name, radius)
                data.setdefault(key, {"coll": [], "succ": []})
                data[key]["coll"].append(float(c[f"r{radius}_coll"]))
                data[key]["succ"].append(float(c[f"r{radius}_succ"]))

    rows = []
    for name, _ in GROUPS:
        for radius in RADII:
            d = data[(name, radius)]
            rows.append([name, radius, float(np.mean(d["coll"])), float(np.std(d["coll"])),
                         float(np.mean(d["succ"])), float(np.std(d["succ"]))])
    with open("full_obs_radius.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "radius", "collision_mean", "collision_std", "success_mean", "success_std", "n_seeds"])
        w.writerows([r + [N_AGG[0]] for r in rows])

    hid = {}
    with open("radius_generalization.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["group"] in ("no_apf", "ours"):
                hid[(r["group"], float(r["radius"]))] = float(r["collision_mean"])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"no_apf": "#d62728", "ours": "#2ca02c"}
    for name, _ in GROUPS:
        xs = RADII
        ys = [float(np.mean(data[(name, r)]["coll"])) for r in RADII]
        es = [float(np.std(data[(name, r)]["coll"])) for r in RADII]
        ax.errorbar(xs, ys, yerr=es, color=colors[name], ls="-", lw=2.5, marker="o",
                    ms=6, capsize=3, label=f"full-obs {name}")
        hy = [hid[(name, r)] for r in RADII]
        ax.plot(xs, hy, color=colors[name], ls="--", lw=2, marker="s", ms=5,
                alpha=0.6, label=f"hidden-obs {name}")
    ax.axvline(TRAIN_RADIUS, color="gray", ls=":", lw=1.5, alpha=0.8)
    ax.set_xlabel("Obstacle radius (m)")
    ax.set_ylabel("Collision rate")
    ax.set_title("full-obs vs hidden-obs: collision rate vs radius (n=%d seeds)" % N_AGG[0])
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = "Paper_Figure_fullobs_vs_hidden.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[已保存] {out} + full_obs_radius.csv")
    for name, _ in GROUPS:
        print(f"full-obs {name}: " + "  ".join(
            f"r={r:.3f}→{np.mean(data[(name,r)]['coll'])*100:.0f}%" for r in RADII))


if __name__ == "__main__":
    main()
