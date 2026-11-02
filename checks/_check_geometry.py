"""Check the geometric statements of the Experimental Setup (straight-line crossing fractions)."""
import numpy as np
import gymnasium as gym
import panda_gym
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
raw.unwrapped.task.distance_threshold = 0.03
e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, "random_walk", radius=0.03),
                         success_threshold=0.03, eval_mode=True, collision_penalty=0.2,
                         step_cost=0.05, hide_obs=True)
starts = []
for s in range(5):
    obs, _ = e.reset(seed=10000 + s)
    starts.append(obs["observation"][:3].copy())
print("start EE positions", np.round(starts, 4))
start = np.array(starts[0], float)

rng = e.unwrapped.task.np_random
A = np.array([-0.07, 0.045])
B = np.array([-0.19, 0.015])
H = 0.21


def seg_dist(p0, p1, c):
    """min horizontal distance from c to the xy-projection of segment p0->p1 (only the part with z<=H)."""
    d = p1[:2] - p0[:2]
    t = np.clip(np.dot(c - p0[:2], d) / np.dot(d, d), 0, 1)
    return np.linalg.norm(p0[:2] + t * d - c)


def stats(goals, rho):
    a = b = both = any_ = 0
    for g in goals:
        ia = seg_dist(start, g, A) < rho
        ib = seg_dist(start, g, B) < rho
        a += ia
        b += ib
        both += ia and ib
        any_ += ia or ib
    n = len(goals)
    return a / n, b / n, both / n, any_ / n


# (1) the 50 evaluation goals
goals = []
for s in range(50):
    e.reset(seed=10000 + s)
    goals.append(e._sample_cross_obstacle_goal().astype(float))
print("50 eval goals     A/B/both/any @0.03:", np.round(stats(goals, 0.03), 3))

# (2) Monte Carlo over the sampling distribution (many goals)
goals = []
for _ in range(20000):
    e.reset()
    goals.append(e._sample_cross_obstacle_goal().astype(float))
for rho in (0.03, 0.055):
    print(f"MC 20000 goals     A/B/both/any @{rho}:", np.round(stats(goals, rho), 3))

# fixed example goal used in the figure
fg = np.array([-0.33, 0.06, 0.10])
print("FIXED_GOAL dist to A,B centres:", seg_dist(start, fg, A), seg_dist(start, fg, B))

# start distance to pillars
print("start dist to A,B", np.linalg.norm(start[:2] - A), np.linalg.norm(start[:2] - B))
# goal region bounds
print("goal x in [-0.38,-0.30], y in [0.03,0.15], z in [0.08,0.13]")
# is any goal sampled such that straight-line segment from start does not cross any pillar?
