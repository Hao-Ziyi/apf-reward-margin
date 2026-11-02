"""实验场景渲染图（PyBullet 真实场景，非示意图）：
Panda 机械臂 + 两根圆柱障碍 + 目标采样区域 + 示例目标。

所有几何均直接取自 multi_obstacle_env.py 的真实环境（柱位、半径、柱高、起点由 env.reset 产生），
不手工摆放。标注由 3D→2D 投影自动定位。用法：
    python render_scene.py            # 只渲染场景  -> Paper_Figure_Scene3D.png
    python render_scene.py rollout    # 叠加真实策略轨迹 -> Paper_Figure_Scene3D_Rollout.png
"""
import sys
import numpy as np
import pybullet as p
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles, FIXED_GOAL

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

W, H = 2400, 1500
RADIUS = 0.03
START_SEED = 10000

# 相机
CAM = dict(yaw=25, pitch=-22, dist=1.34, target=(-0.22, 0.03, 0.24))

# 配色：低饱和、统一色调
C_PILLAR = [0.30, 0.45, 0.62, 1.0]       # 钢蓝
C_TABLE = [0.93, 0.92, 0.90, 1.0]        # 浅米白
C_PLANE = [0.97, 0.97, 0.97, 1.0]
C_GOAL = [0.85, 0.33, 0.10, 1.0]         # 暖橙（目标球 + 采样区域边框）
C_OURS = [0.10, 0.62, 0.47, 1.0]         # 青绿
C_NOAPF = [0.80, 0.25, 0.25, 1.0]        # 砖红
HEX = lambda c: "#%02x%02x%02x" % tuple(int(255 * v) for v in c[:3])


def make_env(radius=RADIUS):
    raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100,
                   render_mode="rgb_array")
    e = MultiObstacleWrapper(raw, obstacles=get_obstacles(True, "random_walk", radius=radius),
                             eval_mode=True, collision_penalty=0.2, step_cost=0.05,
                             hide_obs=True)
    obs, _ = e.reset(seed=START_SEED)
    obs = e._force_goal(obs, FIXED_GOAL)
    return e, obs


def recolor(e):
    c = e.unwrapped.sim.physics_client
    for bid in e.obstacle_ids:
        c.changeVisualShape(bid, -1, rgbaColor=C_PILLAR, specularColor=[0.4, 0.4, 0.4])
    for name, rgba in [("table", C_TABLE), ("plane", C_PLANE)]:
        bid = e.unwrapped.sim._bodies_idx[name]
        for link in range(-1, c.getNumJoints(bid)):
            c.changeVisualShape(bid, link, rgbaColor=rgba, textureUniqueId=-1)
    tid = e.unwrapped.sim._bodies_idx["target"]
    c.changeVisualShape(tid, -1, rgbaColor=C_GOAL)


def add_box_frame(c, lo, hi, rgba, t=0.0016):
    """目标采样区域：12 条细边框（TinyRenderer 不支持 alpha，故用线框而非实心）。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for a in (lo[others[0]], hi[others[0]]):
            for b in (lo[others[1]], hi[others[1]]):
                half = np.full(3, t)
                half[axis] = (hi[axis] - lo[axis]) / 2
                pos = np.zeros(3)
                pos[axis] = (lo[axis] + hi[axis]) / 2
                pos[others[0]], pos[others[1]] = a, b
                vis = c.createVisualShape(p.GEOM_BOX, halfExtents=half.tolist(), rgbaColor=rgba)
                c.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos.tolist())


def add_polyline(c, pts, rgba, r=0.0030, step=0.003):
    """用小球串成轨迹（DIRECT 模式下 debug line 不会被渲染）。"""
    vis = c.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba)
    pts = np.asarray(pts)
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(n):
            q = a + (b - a) * k / n
            c.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=q.tolist())


def rollout(model_path, radius=RADIUS):
    """真实策略 rollout，返回 EE 轨迹（隐观测，确定性策略）。"""
    from stable_baselines3 import PPO
    e, obs = make_env(radius)
    model = PPO.load(model_path)
    goal = FIXED_GOAL.copy()
    traj = [obs["observation"][:3].copy()]
    for _ in range(100):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = e.step(a)
        obs = dict(obs)
        obs["desired_goal"] = goal.copy()
        traj.append(obs["observation"][:3].copy())
        if trunc or np.linalg.norm(obs["observation"][:3] - goal) < 0.03:
            break
    e.close()
    return np.array(traj)


def render_rgb(e):
    c = e.unwrapped.sim.physics_client
    view = c.computeViewMatrixFromYawPitchRoll(list(CAM["target"]), CAM["dist"],
                                               CAM["yaw"], CAM["pitch"], 0, 2)
    proj = c.computeProjectionMatrixFOV(35, W / H, 0.05, 4.0)
    img = c.getCameraImage(W, H, view, proj, shadow=1, lightDirection=[0.4, -0.5, 1.0],
                           renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8)
    return rgb, np.array(view).reshape(4, 4, order="F"), np.array(proj).reshape(4, 4, order="F")


def project(pt, view, proj):
    v = proj @ view @ np.append(np.asarray(pt, float), 1.0)
    ndc = v[:2] / v[3]
    return (ndc[0] + 1) / 2 * W, (1 - ndc[1]) / 2 * H


def annotate(ax, text, xy, xytext, color="#222222", ha="center"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=15, color=color, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=1.4, shrinkA=2, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.2, alpha=0.92))


def build(e, mode, out, ee_start, rollouts=None):
    rgb, view, proj = render_rgb(e)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(rgb)
    ax.axis("off")
    pr = lambda q: project(q, view, proj)
    A, B = e.obstacles[0], e.obstacles[1]
    topA = pr((A["x"], A["y"], A["height"]))
    topB = pr((B["x"], B["y"], B["height"]))
    annotate(ax, "Pillar A", topA, (topA[0] + 170, topA[1] - 110))
    annotate(ax, "Pillar B", topB, (topB[0] - 20, topB[1] - 130))
    annotate(ax, "Start (EE)", pr(ee_start), (pr(ee_start)[0] + 190, pr(ee_start)[1] + 150),
             color="#1a7a3a")
    reg_top = pr((-0.34, 0.15, 0.13))
    annotate(ax, "Goal sampling region", reg_top, (reg_top[0] - 230, reg_top[1] - 120),
             color=HEX(C_GOAL))
    if rollouts:
        # 图例
        for i, (name, col) in enumerate(rollouts):
            ax.plot([], [], "-", color=HEX(col), lw=4, label=name)
        ax.legend(loc="lower left", fontsize=15, frameon=True, facecolor="white", framealpha=0.92)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("[已保存]", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scene"
    e, obs = make_env()
    ee_start = obs["observation"][:3].copy()
    recolor(e)
    c = e.unwrapped.sim.physics_client
    # 目标采样区域 x∈[-0.38,-0.30], y∈[0.03,0.15], z∈[0.08,0.13]（与 env 一致）
    add_box_frame(c, [-0.38, 0.03, 0.08], [-0.30, 0.15, 0.13], C_GOAL)
    if mode == "scene":
        build(e, mode, "Paper_Figure_Scene3D.png", ee_start)
    else:
        rolls = [("no_apf", "models_formal/dynamic_no_apf_seed1.zip", C_NOAPF),
                 ("ours (APF)", "models_formal/dynamic_ours_seed1.zip", C_OURS)]
        for name, path, col in rolls:
            tr = rollout(path)
            print(name, "points", len(tr), "end", tr[-1])
            add_polyline(c, tr, col)
        build(e, mode, "Paper_Figure_Scene3D_Rollout.png", ee_start,
              rollouts=[(n, col) for n, _, col in rolls])


