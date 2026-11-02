import gc
import os
import csv
import argparse
import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import PPO
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

TIMESTEPS = 150000      # 每模型总训练步数（样本效率度量：只够看"谁先学会"，不训到追平）
CHUNK = 15000           # 评估频率：150k / 15k = 10 个评估点（整数）
SUCCESS_THRESHOLD = 0.03
TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_EVAL = 20             # 每块后评估的随机跨障碍目标数

# ---- 任务环境常量（所有组共享，非 shaping）----
COLLISION_PENALTY = 0.2   # 撞柱每步扣分（甜点区 p*）
STEP_COST = 0.05          # 时间惩罚（效率↔安全 trade-off）
MOTION = "random_walk"    # 动态障碍运动：随机游走（不可预测）

MODEL_DIR = "models_formal"
CURVE_DIR = "curves_formal"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)


def make_env():
    env = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    env.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
    return env


# 5 组消融（height 已并入 APF；smooth 固定 0.03 不作为消融变量）：
#   vanilla      = 无任何 shaping（success/progress/safety/height 全 0）
#   no_apf       = 有 success+progress+smooth，无 APF(+height) → 证明 APF 必要性
#   no_progress  = 有 success+APF+smooth，无 progress → 归因 progress(PBRS) 的收敛加速
#   ours         = 全有
#   no_xy_apf    = 只去水平 xy APF(safety_weight=0)，保留 height_penalty=2.0（竖直封顶）
#                  → 归因控制：区分「水平势场」与「竖直封顶」哪个才是 OOD 鲁棒的原因。
GROUPS = [
    ("no_apf", dict(safety_weight=0.0, height_penalty=0.0)),
    ("ours", dict()),
    ("no_progress", dict(progress_weight=0.0)),
    ("vanilla", dict(smoothness_weight=0.0, safety_weight=0.0, height_penalty=0.0,
                     success_bonus=0.0, progress_weight=0.0)),
    ("no_xy_apf", dict(safety_weight=0.0)),
]

# 场景顺序：动态优先（区分度关键，先监控 ours vs no_apf）。
# 只训动态（随机游走）：静态区分度弱（~7%，见 HANDOFF 第六节），留作后续主消融表。
SCENARIOS = [
    ("dynamic", True),
]


def evaluate_model(model, kw, dynamic):
    """块间评估：返回 (成功率, 撞柱率, 安全成功率, 平均路径长)。
    每集记录联合事件 (reached, collided)，安全成功率 = 到达∧无碰撞。"""
    succ = n_coll = n_safe = 0
    path_len = []
    for s in range(N_EVAL):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(raw, obstacles=get_obstacles(dynamic, motion=MOTION),
                                 success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
                                 collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST,
                                 hide_obs=True, **kw)
        obs, _ = e.reset(seed=10000 + s)
        goal = e._sample_cross_obstacle_goal()
        obs = e._force_goal(obs, goal)
        traj = [obs["observation"][:3].copy()]
        ep_coll = False
        t = 0
        for t in range(100):
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
        reached = np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD
        collided = bool(ep_coll)
        if reached:
            succ += 1
            if not collided:
                n_safe += 1
        if collided:
            n_coll += 1
        P = np.array(traj)
        path_len.append(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    return succ / N_EVAL, n_coll / N_EVAL, n_safe / N_EVAL, float(np.mean(path_len))


def train_one(scenario, dynamic, name, kw, seed):
    """训练单模型，返回学习曲线 [(ts, success, collision, path)]。"""
    env = MultiObstacleWrapper(
        make_env(), obstacles=get_obstacles(dynamic, motion=MOTION),
        success_threshold=SUCCESS_THRESHOLD,
        collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST,
        hide_obs=True, **kw)
    model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=0.01)

    curve = []
    n_chunks = TIMESTEPS // CHUNK
    for i in range(n_chunks):
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(i == 0))
        ts = (i + 1) * CHUNK
        sr, cr, sfr, pl = evaluate_model(model, kw, dynamic)
        curve.append((ts, sr, cr, sfr, pl))

    model.save(f"{MODEL_DIR}/{scenario}_{name}_seed{seed}")
    env.close()
    return model, curve


def _already_done(tag):
    """训练恢复点：若该模型的 CSV 已是新格式（含 safe_success_rate 列），视为已完成，跳过。
    旧 6-seed 格式（无 safe_success_rate 列）不会被跳过 → 会被 10-seed 重训覆盖。"""
    csv_path = os.path.join(CURVE_DIR, f"{tag}.csv")
    if not os.path.exists(csv_path):
        return False
    with open(csv_path) as f:
        header = f.readline()
    return "safe_success_rate" in header


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=None,
                   help="只训这些组（组名），默认全部；如 --groups ours no_apf")
    p.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS,
                   help="只训这些 seed（默认 1..10）；补种子用 --seeds 11 12 ... 30")
    args = p.parse_args()
    groups = GROUPS if args.groups is None else [g for g in GROUPS if g[0] in args.groups]
    seeds = args.seeds

    print("\n" + "=" * 80)
    print(f"【正式训练 · {len(groups)} 组 × 1 场景 × {len(seeds)} seed × 150k · p={COLLISION_PENALTY} sc={STEP_COST} motion={MOTION}】")
    print("=" * 80)
    for scenario, dynamic in SCENARIOS:
        for name, kw in groups:
            for seed in seeds:
                tag = f"{scenario}_{name}_seed{seed}"
                if _already_done(tag):
                    print(f"⏭ 跳过 {tag}（已完成，新格式）", flush=True)
                    continue
                print(f"\n>>> 训练 {tag} ...", flush=True)
                model, curve = train_one(scenario, dynamic, name, kw, seed)

                with open(f"{CURVE_DIR}/{tag}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["timestep", "success_rate", "collision_rate", "safe_success_rate", "path_len"])
                    w.writerows(curve)

                print(f"✓ {tag}  成功率={curve[-1][1]:.1%} 撞柱率={curve[-1][2]:.1%} 安全成功={curve[-1][3]:.1%} 路径={curve[-1][4]:.3f}", flush=True)
                del model
                gc.collect()
    print("\n" + "=" * 80)
    print("★ 正式训练完成！")
    print(f"  模型在 {MODEL_DIR}/，学习曲线在 {CURVE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
