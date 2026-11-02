"""Control: inflated collision radius (hard-edged penalty zone) instead of the APF potential.

Trains the `no_apf` reward (no horizontal potential, no height penalty) but charges the per-step
collision penalty P whenever the end-effector is inside rho + m of a pillar (m = inflation, in the
pillar height band).  Only the *training reward* is inflated; evaluation uses the true radius and the
true collision indicator, exactly as in verify_radius_generalize.py.

    python pilot_inflate.py --inflation 0.05 --penalty 0.2 --seeds 1 2

Outputs: models_inflate/no_apf_m{mm}_p{P}_seed{s}.zip, inflate_cache/no_apf_m{mm}_p{P}_seed{s}.npz
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
GROUP = "no_apf"
KW = dict(V.GROUPS)[GROUP]

MODEL_DIR = "models_inflate"
CACHE_DIR = "inflate_cache"


def train(inflation, penalty, seed):
    raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
    env = MultiObstacleWrapper(
        raw, obstacles=get_obstacles(True, motion=MOTION), success_threshold=SUCCESS_THRESHOLD,
        collision_penalty=penalty, step_cost=STEP_COST, hide_obs=True,
        collision_inflation=inflation, **KW)
    model = PPO("MultiInputPolicy", env, seed=seed, verbose=0, ent_coef=0.01)
    model.learn(total_timesteps=TIMESTEPS)
    env.close()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inflation", type=float, required=True, help="m, in meters (e.g. 0.05)")
    ap.add_argument("--penalty", type=float, default=0.2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    goals = V.sample_goals()
    for seed in args.seeds:
        tag = f"{GROUP}_m{args.inflation * 1000:g}_p{args.penalty:g}_seed{seed}"
        cache_path = f"{CACHE_DIR}/{tag}.npz"
        if os.path.exists(cache_path):
            print("[skip]", tag, flush=True)
            continue
        print(">>> training", tag, flush=True)
        model = train(args.inflation, args.penalty, seed)
        model.save(f"{MODEL_DIR}/{tag}")
        cache = {}
        for radius in V.RADII:
            r = V.run_model(model, KW, radius, goals)
            for k, v in r.items():
                cache[f"r{radius}_{k}"] = v
            print(f"  {tag} r={radius:.3f} coll={r['coll']*100:5.1f}% safe={r['safe']*100:5.1f}% "
                  f"clear={r['clear']*1000:6.1f}mm path={r['path']:.3f}", flush=True)
        np.savez(cache_path, **cache)


if __name__ == "__main__":
    main()
