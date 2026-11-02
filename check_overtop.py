"""审稿人质疑核验：「为什么不翻顶？翻顶是不是回报更高？」—— 用真实环境 + 真实奖励回答。

方法：
  1) 对每个目标，在参数网格上搜索「翻顶」航点控制器（升到柱顶以上 → 平移越过两柱 → 下降到目标，
     全速 per-axis 控制），取各参数中回报最高且无碰撞的一条 = 翻顶路线的（近似）最优回报。
  2) 与学到的低位策略（PPO seed1..5，确定性）在**同一批目标、同一奖励配置、同一随机游走种子**下比较
     回报、到达步数、撞柱率。
  3) 回报 = env 实际回报（环境在到达目标时 terminated，回报即 episode 总和，与训练一致）。
奖励配置取 no_apf / vanilla（无高度惩罚，翻顶不受罚）与 ours（有高度惩罚）。
结论仅限于「该参数化翻顶路线族」，不是对所有翻顶轨迹的最优性证明。
输出打印到终端。
"""
import sys
import numpy as np
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

CFGS = {
    "no_apf": dict(safety_weight=0.0, height_penalty=0.0),
    "vanilla": dict(smoothness_weight=0.0, safety_weight=0.0, height_penalty=0.0,
                    success_bonus=0.0, progress_weight=0.0),
    "ours": dict(),
}
N_GOALS = 12
SEEDS = [1, 2, 3, 4, 5]
ZC_GRID = [0.213, 0.216, 0.222]   # 翻顶巡航高度（柱顶 0.21 之上 3/6/12 mm）
X1_GRID = [0.035, 0.025, 0.015, 0.005, -0.005]  # 在此 x 之前必须已升到 ZC（柱 A 足迹最右端 ρ=0.055 时 ≈ 0.0）
X2_GRID = [-0.21, -0.23, -0.25, -0.27, -0.29]  # 越过柱 B 足迹后开始下降


def make_env(kw, radius):
    raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
    raw.unwrapped.task.distance_threshold = 0.03
    return MultiObstacleWrapper(raw, obstacles=get_obstacles(True, "random_walk", radius=radius),
                                success_threshold=0.03, eval_mode=True, collision_penalty=0.2,
                                step_cost=0.05, hide_obs=True, **kw)


def sample_goals(n):
    e = make_env({}, 0.03)
    goals = []
    for s in range(n):
        e.reset(seed=10000 + s)
        goals.append(e._sample_cross_obstacle_goal().astype(float))
    e.close()
    return goals


def episode(e, seed, goal, policy):
    obs, _ = e.reset(seed=seed)
    obs = e._force_goal(obs, goal)
    ret, coll_steps, zmax = 0.0, 0, 0.0
    start = obs["observation"][:3].copy()
    policy.reset(start, goal)
    for t in range(100):
        a = policy.act(obs)
        obs, r, term, trunc, info = e.step(a)
        obs = dict(obs)
        obs["desired_goal"] = np.asarray(goal, dtype=np.float32).copy()
        ret += r
        coll_steps += int(info.get("is_collision", False))
        zmax = max(zmax, float(obs["observation"][2]))
        if term or trunc:
            break
    reached = bool(np.linalg.norm(obs["observation"][:3] - goal) < 0.03)
    return dict(ret=ret, steps=t + 1, coll=coll_steps > 0, reached=reached, zmax=zmax)


class Learned:
    def __init__(self, path):
        self.m = PPO.load(path)

    def reset(self, start, goal):
        pass

    def act(self, obs):
        return self.m.predict(obs, deterministic=True)[0]


class OverTop:
    def __init__(self, x1, x2, zc):
        self.x1, self.x2, self.zc = x1, x2, zc

    def reset(self, start, goal):
        self.start, self.goal, self.phase = np.asarray(start, float), np.asarray(goal, float), 0

    def _y(self, x):
        s, g = self.start, self.goal
        return s[1] + (g[1] - s[1]) * (x - s[0]) / (g[0] - s[0])

    def act(self, obs):
        ee = np.asarray(obs["observation"][:3], float)
        if self.phase == 0 and ee[2] >= self.zc - 0.004:
            self.phase = 1
        if self.phase == 1 and ee[0] <= self.x2 + 0.012:
            self.phase = 2
        if self.phase == 0:
            tgt = np.array([self.x1, self._y(self.x1), self.zc])
        elif self.phase == 1:
            tgt = np.array([self.x2, self._y(self.x2), self.zc])
        else:
            tgt = self.goal
        return np.clip((tgt - ee) / 0.05, -1, 1).astype(np.float32)


def main():
    goals = sample_goals(N_GOALS)
    print(f"goals={N_GOALS}, learned seeds={SEEDS}, over-top grid x1={X1_GRID} x2={X2_GRID} zc={ZC_GRID}")
    hdr = f"{'cfg':<8}{'rho':>6} {'route':<24}{'return':>8}{'±std':>7}{'steps':>7}{'coll%':>7}{'reach%':>8}"
    print(hdr)
    for cfg, kw in CFGS.items():
        for rho in (0.03, 0.055):
            e = make_env(kw, rho)
            # --- 翻顶：每目标取网格内最优（必须无碰撞）
            best = []
            for i, g in enumerate(goals):
                cands = []
                for x1 in X1_GRID:
                    for x2 in X2_GRID:
                        for zc in ZC_GRID:
                            r = episode(e, 10000 + i, g, OverTop(x1, x2, zc))
                            if r["reached"] and not r["coll"]:
                                cands.append(r)
                best.append(max(cands, key=lambda r: r["ret"]) if cands else None)
            ok = [b for b in best if b is not None]
            ret = np.array([b["ret"] for b in ok])
            print(f"{cfg:<8}{rho:>6} {'over-top (best of grid)':<24}{ret.mean():>8.2f}{ret.std():>7.2f}"
                  f"{np.mean([b['steps'] for b in ok]):>7.1f}{0.0:>7.0f}{100*len(ok)/N_GOALS:>8.0f}")
            # --- 学到的低位策略（逐 seed 汇总）
            per_goal_learned = np.zeros(N_GOALS)
            rets, steps, colls, reach = [], [], [], []
            for sd in SEEDS:
                pol = Learned(f"models_formal/dynamic_{cfg}_seed{sd}.zip")
                sret = []
                for i, g in enumerate(goals):
                    r = episode(e, 10000 + i, g, pol)
                    sret.append(r["ret"])
                    rets.append(r["ret"]); steps.append(r["steps"])
                    colls.append(r["coll"]); reach.append(r["reached"])
                per_goal_learned += np.array(sret) / len(SEEDS)
            rets = np.array(rets)
            print(f"{cfg:<8}{rho:>6} {'learned low-level (PPO)':<24}{rets.mean():>8.2f}{rets.std():>7.2f}"
                  f"{np.mean(steps):>7.1f}{100*np.mean(colls):>7.0f}{100*np.mean(reach):>8.0f}")
            # 逐目标配对：翻顶最优是否高于学到的策略（仅对有可行翻顶解的目标）
            wins = [best[i]["ret"] > per_goal_learned[i] for i in range(N_GOALS) if best[i] is not None]
            print(f"{'':<14}over-top beats learned on {sum(wins)}/{len(wins)} goals (paired, seed-avg)\n")
            e.close()


if __name__ == "__main__":
    main()


