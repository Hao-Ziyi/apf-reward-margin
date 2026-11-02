"""求解器稳健性检验（P0）的共享工具：把 Dict 观测展平成单一 Box。

动机：SB3 的 SAC / TD3 只支持 Box（扁平）观测空间，不支持 Dict（PPO 的 MultiInputPolicy
才支持）。而 PandaReach-v3 的观测是 Dict('observation'(6), 'desired_goal'(3), 'achieved_goal'(3))。

做法：固定展平顺序 [observation, desired_goal, achieved_goal]，训练用 FlatObsWrapper 包装
环境（提供 Box 观测），评估用 flatten_obs() 就地展平后喂给 model.predict —— 两处顺序一致。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def flatten_obs(obs):
    """Dict 观测 → 一维 float32 向量，顺序 [observation, desired_goal, achieved_goal]。"""
    return np.concatenate([
        np.asarray(obs["observation"], dtype=np.float32).ravel(),
        np.asarray(obs["desired_goal"], dtype=np.float32).ravel(),
        np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
    ]).astype(np.float32)


class FlatObsWrapper(gym.Wrapper):
    """把 MultiObstacleWrapper 的 Dict 观测包装成单一 Box（供 SAC/TD3 训练）。"""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space.spaces["observation"]
        dg_space = env.observation_space.spaces["desired_goal"]
        ag_space = env.observation_space.spaces["achieved_goal"]
        self._obs_dim = int(obs_space.shape[0])
        self._dg_dim = int(dg_space.shape[0])
        low = np.concatenate([obs_space.low, dg_space.low, ag_space.low]).astype(np.float32)
        high = np.concatenate([obs_space.high, dg_space.high, ag_space.high]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return flatten_obs(obs), reward, terminated, truncated, info
