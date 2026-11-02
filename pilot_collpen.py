"""Pilot: does a larger per-step collision penalty alone produce a size-robust margin?

Trains the `no_apf` reward (no horizontal potential, no height penalty) and, optionally, `ours`
with a collision penalty P > 0.2 (default 0.2), hidden observation, PPO, fixed training radius 0.03,
then evaluates on the usual radius sweep with the same goals as verify_radius_generalize.py.

    python pilot_collpen.py --penalty 1.0 --groups no_apf --seeds 1 2

Outputs: models_collpen/{group}_p{P}_seed{s}.zip, collpen_cache/{group}_p{P}_seed{s}.npz
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
MOTION = "random_walk"
GROUPS = dict(V.GROUPS)

MODEL_DIR = "models_collpen"
CACHE_DIR = "collpen_cache"


def train(group, penalty, seed):
    raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
    env = MultiObstacleWrapper(
        raw, obstacles=get_obstacles(True, motion=MOTION), success_threshold=SUCCESS_THRESHOLD,
        collision_penalty=penalty, step_cost=STEP_COST, hide_obs=True, **GROUPS[group])
    model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=0.01)
    model.learn(total_timesteps=TIMESTEPS)
    env.close()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", type=float, required=True)
    ap.add_argument("--groups", nargs="+", default=["no_apf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    goals = V.sample_goals()
    for group in args.groups:
        for seed in args.seeds:
            tag = f"{group}_p{args.penalty:g}_seed{seed}"
            cache_path = f"{CACHE_DIR}/{tag}.npz"
            if os.path.exists(cache_path):
                print("[skip]", tag, flush=True)
                continue
            print(">>> training", tag, flush=True)
            model = train(group, args.penalty, seed)
            model.save(f"{MODEL_DIR}/{tag}")
            cache = {}
            for radius in V.RADII:
                r = V.run_model(model, GROUPS[group], radius, goals)
                for k, v in r.items():
                    cache[f"r{radius}_{k}"] = v
                print(f"  {tag} r={radius:.3f} coll={r['coll']*100:5.1f}% safe={r['safe']*100:5.1f}% "
                      f"clear={r['clear']*1000:6.1f}mm path={r['path']:.3f}", flush=True)
            np.savez(cache_path, **cache)


if __name__ == "__main__":
    main()
