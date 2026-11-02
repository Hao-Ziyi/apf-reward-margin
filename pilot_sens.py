"""Sensitivity of the APF reward to d_safe and lambda_s (one-at-a-time around the defaults).

Trains `ours` (full reward) with a modified safety margin d_safe (m) and/or potential weight
lambda_s, PPO, hidden observation, training radius 0.03, 150k steps; evaluates on the usual radius
sweep (same 50 goals as verify_radius_generalize.py).

    python pilot_sens.py --dsafe 0.06 --lam 1.5 --seeds 1 2 3

Outputs: models_sens/ours_d{mm}_l{lam}_seed{s}.zip, sens_cache/ours_d{mm}_l{lam}_seed{s}.npz
"""
import argparse
import os
import numpy as np
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles
import verify_radius_generalize as V

TIMESTEPS = 150000
SUCCESS_THRESHOLD = 0.03
STEP_COST = 0.05
COLLISION_PENALTY = 0.2
MOTION = "random_walk"

MODEL_DIR = "models_sens"
CACHE_DIR = "sens_cache"


def train(dsafe, lam, seed):
    raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
    env = MultiObstacleWrapper(
        raw, obstacles=get_obstacles(True, motion=MOTION), success_threshold=SUCCESS_THRESHOLD,
        collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST, hide_obs=True,
        safe_margin=dsafe, safety_weight=lam)
    model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=0.01)
    model.learn(total_timesteps=TIMESTEPS)
    env.close()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsafe", type=float, default=0.08)
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    goals = V.sample_goals()
    for seed in args.seeds:
        tag = f"ours_d{args.dsafe * 1000:g}_l{args.lam:g}_seed{seed}"
        cache_path = f"{CACHE_DIR}/{tag}.npz"
        if os.path.exists(cache_path):
            print("[skip]", tag, flush=True)
            continue
        print(">>> training", tag, flush=True)
        model = train(args.dsafe, args.lam, seed)
        model.save(f"{MODEL_DIR}/{tag}")
        cache = {}
        for radius in V.RADII:
            r = V.run_model(model, {}, radius, goals)
            for k, v in r.items():
                cache[f"r{radius}_{k}"] = v
            print(f"  {tag} r={radius:.3f} coll={r['coll']*100:5.1f}% safe={r['safe']*100:5.1f}% "
                  f"clear={r['clear']*1000:6.1f}mm path={r['path']:.3f}", flush=True)
        np.savez(cache_path, **cache)


if __name__ == "__main__":
    main()
