"""Two extra checks on claims in experiments.tex:
 (1) Is the classical-APF failure a stall (equilibrium) in the corridor between the pillars?
 (2) Is the ceiling penalty H ever active (z > h anywhere along trajectories) for the learned policies?
"""
import numpy as np
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
import classic_apf_baseline as cb
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

A = np.array([-0.07, 0.045])
B = np.array([-0.19, 0.015])

# ---------------- (1) classical APF stall analysis @ rho = 0.03
goals = cb.sample_goals()
res = []
for s, goal in enumerate(goals):
    raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    raw.unwrapped.task.distance_threshold = 0.03
    e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, "random_walk", radius=0.03),
                             success_threshold=0.03, eval_mode=True, collision_penalty=0.2,
                             step_cost=0.05, hide_obs=True)
    obs, _ = e.reset(seed=10000 + s)
    obs = e._force_goal(obs, goal)
    traj = [obs["observation"][:3].copy()]
    for _ in range(100):
        ee = obs["observation"][:3]
        a = cb.apf_action(ee, goal, e.obstacles)
        obs, r, term, trunc, info = e.step(a)
        obs = dict(obs)
        obs["desired_goal"] = goal.copy()
        traj.append(obs["observation"][:3].copy())
        if trunc or np.linalg.norm(obs["observation"][:3] - goal) < 0.03:
            break
    e.close()
    P = np.array(traj)
    reached = np.linalg.norm(P[-1] - goal) < 0.03
    last = P[-20:] if len(P) >= 20 else P
    net = np.linalg.norm(last[-1] - last[0])
    path = np.sum(np.linalg.norm(np.diff(last, axis=0), axis=1))
    res.append((reached, np.linalg.norm(P[-1] - goal), P[-1][:2], net, path, len(P) - 1,
                np.linalg.norm(P[-1][:2] - A) - 0.03, np.linalg.norm(P[-1][:2] - B) - 0.03))
stuck = [r for r in res if not r[0]]
print(f"classic APF rho=0.03: reached {sum(r[0] for r in res)}/50, not reached {len(stuck)}")
print("  final dist to goal (m): median %.3f, range %.3f-%.3f" % (np.median([r[1] for r in stuck]),
      min(r[1] for r in stuck), max(r[1] for r in stuck)))
print("  last-20-step net displacement (m): median %.4f ; path in same window: median %.4f" % (
      np.median([r[3] for r in stuck]), np.median([r[4] for r in stuck])))
print("  final clearance to A / B (mm): median %.1f / %.1f" % (
      1000 * np.median([r[6] for r in stuck]), 1000 * np.median([r[7] for r in stuck])))
fx = np.array([r[2] for r in stuck])
print("  final xy median:", np.median(fx, axis=0), " x range", fx[:, 0].min(), fx[:, 0].max())

# ---------------- (2) max z for learned policies, hidden obs (trajectory independent of rho)
kws = {"ours": dict(), "no_apf": dict(safety_weight=0.0, height_penalty=0.0),
       "no_xy_apf": dict(safety_weight=0.0), "vanilla": dict(smoothness_weight=0.0, safety_weight=0.0,
       height_penalty=0.0, success_bonus=0.0, progress_weight=0.0), "no_progress": dict(progress_weight=0.0)}
for name, kw in kws.items():
    n_ep = n_above = 0
    zmax_all = []
    for seed in range(1, 11):
        model = PPO.load(f"models_formal/dynamic_{name}_seed{seed}.zip")
        for s, goal in enumerate(goals):
            raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
            raw.unwrapped.task.distance_threshold = 0.03
            e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, "random_walk", radius=0.03),
                                     success_threshold=0.03, eval_mode=True, collision_penalty=0.2,
                                     step_cost=0.05, hide_obs=True, **kw)
            obs, _ = e.reset(seed=10000 + s)
            obs = e._force_goal(obs, goal)
            zmax = float(obs["observation"][2])
            for _ in range(100):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = e.step(a)
                obs = dict(obs)
                obs["desired_goal"] = goal.copy()
                zmax = max(zmax, float(obs["observation"][2]))
                if trunc or np.linalg.norm(obs["observation"][:3] - goal) < 0.03:
                    break
            e.close()
            n_ep += 1
            n_above += zmax > 0.21
            zmax_all.append(zmax)
    print(f"{name:<12} seeds 1-10 x 50 goals: episodes with z>0.21 anywhere = {n_above}/{n_ep}; max z = {max(zmax_all):.4f}")
