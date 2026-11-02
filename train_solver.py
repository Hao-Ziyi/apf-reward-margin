"""求解器稳健性检验（P0）：换求解器（SAC / TD3）训「ours」组（APF shaping）。

目的：堵审稿人「换个 RL 算法，APF 的 OOD 鲁棒性还成立吗？」。
- 复用 train_multi.py 的环境封装（MultiObstacleWrapper，hide_obs=True，随机游走），
  只把 PPO → SAC / TD3，其余协议完全一致（150k 步、每 15k 评估、10 seed）。
- SAC/TD3 只支持 Box 观测，故用 solver_common.FlatObsWrapper 展平 Dict 观测；
  评估用 flatten_obs() 就地展平（与训练同顺序），复用 train_multi 的 evaluate_model 口径。

定位：SAC/TD3 是「换求解器的稳健性检验」，不是「独立 baseline 方法」。
三求解器覆盖 on/off-policy × 随机/确定性 四象限三个：PPO(on+随机)、SAC(off+随机)、TD3(off+确定性)。

产出：models_solver/{SAC,TD3}_{ours,no_apf}_seed{1..10}.zip + curves_solver/ 同名 .csv
（ours 已训 10 seed；no_apf 按需补训，与 PPO 主实验叠图回答「APF 优势是否求解器无关」）
"""
import argparse
import gc
import os
import csv
import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles
from solver_common import FlatObsWrapper, flatten_obs

TIMESTEPS = 150000      # 与 PPO 一致：只够看「谁先学会」，不训到追平
CHUNK = 15000           # 评估频率：150k / 15k = 10 个评估点
SUCCESS_THRESHOLD = 0.03
TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_EVAL = 20             # 每块后评估的随机跨障碍目标数

# ---- 任务环境常量（与 train_multi.py 完全一致）----
COLLISION_PENALTY = 0.2
STEP_COST = 0.05
MOTION = "random_walk"

MODEL_DIR = "models_solver"
CURVE_DIR = "curves_solver"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

# 两组消融（与 train_multi.py 的 GROUPS 对齐，便于和 PPO 主实验叠图）：
#   ours    = 全 shaping（APF + progress + smooth + success + height）
#   no_apf  = 有 success+progress+smooth，无 APF(+height) → 证明「换求解器后 APF 必要性仍成立」
GROUPS = {
    "ours": dict(),
    "no_apf": dict(safety_weight=0.0, height_penalty=0.0),
}

# 重训补丁：TD3 train_freq 默认 2（与既有 10 seed 一致）；坏种子重训时用 --td3-train-freq 1 提高稳定性。
TD3_TRAIN_FREQ = 2

# 超参说明（CPU 实用性，对 12 维 obs / 3 维 act / 100 步小任务属于合理配置）：
#   - net_arch 缩小（SAC [64,64] / TD3 [128,128]）：SB3 默认 [256,256]/[400,300] 是为 MuJoCo 规模调的，本任务偏小。
#   - batch_size 64：小任务下梯度估计已足够稳。
#   - train_freq=2：每 2 步做 1 次梯度更新（off-policy 常规加速手段，非奖励/环境改动）。
#   其余全部 SB3 默认（SAC ent_coef='auto'，TD3 policy_delay=2 等），保持「off-the-shelf 求解器」定位。
SOLVERS = {
    "SAC": lambda env, seed: SAC(
        "MlpPolicy", env, seed=seed, verbose=0,
        train_freq=2, batch_size=64, policy_kwargs=dict(net_arch=[64, 64])),
    "TD3": lambda env, seed: TD3(
        "MlpPolicy", env, seed=seed, verbose=0,
        train_freq=TD3_TRAIN_FREQ, batch_size=64, policy_kwargs=dict(net_arch=[128, 128]),
        # 确定性策略探索不足（默认 sigma=0.1 太小），加大收集期动作噪声，否则无法学会绕行
        action_noise=NormalActionNoise(mean=np.zeros(3), sigma=0.3 * np.ones(3))),
}


def make_env():
    env = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    env.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
    return env


def evaluate_model(model, kw, dynamic):
    """块间评估（与 train_multi.py 同口径）：返回 (成功率, 撞柱率, 安全成功率, 平均路径长)。
    观测就地展平后喂给 SAC/TD3 的 model.predict。"""
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
        for t in range(100):
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


def train_one(solver_name, name, kw, seed):
    """训练单模型（SAC/TD3），返回学习曲线 [(ts, success, collision, safe_success, path)]。"""
    base = MultiObstacleWrapper(
        make_env(), obstacles=get_obstacles(True, motion=MOTION),
        success_threshold=SUCCESS_THRESHOLD,
        collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST,
        hide_obs=True, **kw)
    env = FlatObsWrapper(base)
    model = SOLVERS[solver_name](env, seed)

    curve = []
    n_chunks = TIMESTEPS // CHUNK
    for i in range(n_chunks):
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(i == 0))
        ts = (i + 1) * CHUNK
        sr, cr, sfr, pl = evaluate_model(model, kw, True)
        curve.append((ts, sr, cr, sfr, pl))
        print(f"    {solver_name} seed{seed} @{ts:>6}: 成功={sr:.1%} 撞柱={cr:.1%} "
              f"安全成功={sfr:.1%} 路径={pl:.3f}", flush=True)

    model.save(f"{MODEL_DIR}/{solver_name}_{name}_seed{seed}")
    env.close()
    return model, curve


def _already_done(tag):
    """训练恢复点：CSV 存在且最后一行 timestep 达到目标 TIMESTEPS 才视为完成。
    只写到中途的 CSV（如 probe 短训）不会被跳过，会被完整重训覆盖。"""
    csv_path = os.path.join(CURVE_DIR, f"{tag}.csv")
    if not os.path.exists(csv_path):
        return False
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2 or "safe_success_rate" not in lines[0]:
        return False
    last_ts = int(float(lines[-1].split(",")[0]))
    return last_ts >= TIMESTEPS


def main(args):
    print("\n" + "=" * 80)
    print(f"【求解器稳健性检验 · {args.solvers} × {args.groups} × {len(args.seeds)} seed × "
          f"{args.timesteps} 步 · hidden obs】")
    print("=" * 80)
    for solver_name in args.solvers:
        for group in args.groups:
            for seed in args.seeds:
                tag = f"{solver_name}_{group}_seed{seed}"
                if args.force:
                    for p in (f"{MODEL_DIR}/{tag}.zip", f"{CURVE_DIR}/{tag}.csv"):
                        if os.path.exists(p):
                            os.remove(p)
                            print(f"  [force] 已删除 {p}", flush=True)
                if _already_done(tag):
                    print(f"⏭ 跳过 {tag}（已完成）", flush=True)
                    continue
                print(f"\n>>> 训练 {tag} ...", flush=True)
                model, curve = train_one(solver_name, group, GROUPS[group], seed)

                with open(f"{CURVE_DIR}/{tag}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["timestep", "success_rate", "collision_rate", "safe_success_rate", "path_len"])
                    w.writerows(curve)

                print(f"✓ {tag}  成功率={curve[-1][1]:.1%} 撞柱率={curve[-1][2]:.1%} "
                      f"安全成功={curve[-1][3]:.1%} 路径={curve[-1][4]:.3f}", flush=True)
                del model
                gc.collect()
    print("\n" + "=" * 80)
    print("★ 求解器稳健性检验训练完成！")
    print(f"  模型在 {MODEL_DIR}/，学习曲线在 {CURVE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solvers", nargs="+", default=["SAC", "TD3"])
    p.add_argument("--groups", nargs="+", default=["ours"], choices=["ours", "no_apf"])
    p.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--force", action="store_true", help="重训前删除已有模型/CSV（强制重跑）")
    p.add_argument("--td3-train-freq", type=int, default=TD3_TRAIN_FREQ,
                   help="TD3 训练频率（默认 2；坏种子重训用 1）")
    args = p.parse_args()
    # 允许 --timesteps 覆盖后重算 chunk 数（评估频率保持 15k）
    if args.timesteps != TIMESTEPS:
        globals()["TIMESTEPS"] = args.timesteps
    globals()["TD3_TRAIN_FREQ"] = args.td3_train_freq
    main(args)
