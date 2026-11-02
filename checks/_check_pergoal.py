"""Per-goal reach rate of ours / no_apf (PPO, 30 seeds, 50 evaluation goals, rho=0.03).
Hidden observation => the EE trajectory does not depend on rho, so one radius is enough for reach statistics."""
import sys
import numpy as np
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
import classic_apf_baseline as cb
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

group = sys.argv[1]
kw = dict() if group == "ours" else dict(safety_weight=0.0, height_penalty=0.0)
goals = cb.sample_goals()
reach = np.zeros((30, 50))
for seed in range(1, 31):
    model = PPO.load(f"models_formal/dynamic_{group}_seed{seed}.zip")
    for s, goal in enumerate(goals):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = 0.03
        e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, "random_walk", radius=0.03),
                                 success_threshold=0.03, eval_mode=True, collision_penalty=0.2,
                                 step_cost=0.05, hide_obs=True, **kw)
        obs, _ = e.reset(seed=10000 + s)
        obs = e._force_goal(obs, goal)
        for _ in range(100):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = e.step(a)
            obs = dict(obs); obs["desired_goal"] = goal.copy()
            if trunc or np.linalg.norm(obs["observation"][:3] - goal) < 0.03:
                break
        e.close()
        reach[seed - 1, s] = np.linalg.norm(obs["observation"][:3] - goal) < 0.03
np.save(f"_pergoal_{group}.npy", reach)
gx = np.array([g[0] for g in goals])
rate = reach.mean(0)
print(group, "mean reach", reach.mean())
order = np.argsort(rate)
print("worst 10 goals (idx, x, y, z, reach rate):")
for i in order[:10]:
    print(f"  {i:2d} x={goals[i][0]:.3f} y={goals[i][1]:.3f} z={goals[i][2]:.3f} rate={rate[i]:.2f}")
print("corr(goal x, reach rate) =", np.corrcoef(gx, rate)[0, 1])
print("reach rate for goals x<-0.355:", rate[gx < -0.355].mean(), " n=", (gx < -0.355).sum(), " ; x>=-0.355:", rate[gx >= -0.355].mean())
