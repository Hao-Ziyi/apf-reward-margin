"""经典 APF（人工势场）规划基线 —— 非学习对照（论文泛化面板第 5 条曲线）。

目的：证明「APF 作为解析规划器直接跑」≠「APF 作为 reward shaping 让 RL 学到」。
- 经典 APF 有解析几何特权（直接读柱心位置 `e.obstacles`），等价全观测；
- （2026-10-30 更正）旧版「卡在两柱间局部极小」是步长/预算人为限制所致；修订后见 apf_action 注释与 control 子命令。
- 协议对齐：同 50 随机目标、同 6 半径扫描、train radius 0.03 调参后固定扫全区间（等价 OOD）。

设计（已与用户确认）：
- xy 平面 APF（引力 + Khatib 斥力），z 朝 goal 线性插值。
- 允许翻顶（z 自由；goal z 在中低部 0.08~0.13，实际走低空绕行）。

用法：
  python classic_apf_baseline.py tune    # 只跑 train radius 0.03，打印撞柱/卡住率，用于调参
  python classic_apf_baseline.py         # 全 6 半径扫描 + CSV + 图

产出：classic_apf_radius.csv + Paper_Figure_ClassicAPF.png
"""
import os
import csv
import sys
import gymnasium as gym
import panda_gym
import numpy as np
import matplotlib.pyplot as plt
from multi_obstacle_env import MultiObstacleWrapper, get_obstacles

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SUCCESS_THRESHOLD = 0.03
COLLISION_PENALTY = 0.2
STEP_COST = 0.05
MOTION = "random_walk"
TRAIN_RADIUS = 0.03
RADII = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
N_EVAL = 50

# ---- APF 参数（在 train radius 0.03 上调到 0 撞柱后固定，扫全区间不重调）----
# 2026-10-30 修订（重要）：旧版把合力归一化后乘固定 ACTION_SCALE=0.3（每步 0.015 m），
#   ① 100 步预算内走不完（路径 1.05 m ≈ 100 步 × 0.0105 m），论文里「卡在柱间局部极小」实为步长/预算人为限制；
#   ② 同一规划器给 400 步或放大步长即可到达 46/50 目标且 0 碰撞。
# 新版：引力随误差线性增长（K_ATT 单位 action·m^-1，0.05 m 误差即满幅），合力超过单位模长才饱和，
#   故接近目标时自然减速、可收敛进 0.03 m 成功阈值；动作量程与 RL 相同（[-1,1]^3 × 0.05 m）。
# 旧结果备份：classic_apf_radius_constant_step.csv。
K_ATT = 20.0         # 引力增益（action 每米误差；0.05 m 误差 → 满幅动作）
K_REP = 5.0          # 斥力增益（Khatib）
RHO0 = 0.08          # 斥力影响半径（= RL 的 safe_margin，对齐公平）
K_Z = 20.0           # z 朝 goal 线性插值增益（与引力同量纲）


def apf_action(ee, goal, obstacles):
    """经典 APF：xy 平面引力+斥力，z 独立朝 goal 线性插值。合力超过单位模长时饱和为单位向量。
    返回 [-1,1]^3 的增量 action。"""
    fx = K_ATT * (goal[0] - ee[0])
    fy = K_ATT * (goal[1] - ee[1])
    for oc in obstacles:
        dc = float(np.hypot(ee[0] - oc["x"], ee[1] - oc["y"]))
        rho = dc - oc["radius"]
        # 只在柱高带内且进入影响半径时排斥（翻顶 z>height 不介入）
        if 0.0 <= ee[2] <= oc["height"] and 0.0 < rho < RHO0:
            f_rep = K_REP * (1.0 / rho - 1.0 / RHO0) / (rho ** 2)
            fx += f_rep * (ee[0] - oc["x"]) / dc
            fy += f_rep * (ee[1] - oc["y"]) / dc
    dz = K_Z * (goal[2] - ee[2])
    f = np.array([fx, fy, dz], dtype=np.float32)
    n = float(np.linalg.norm(f))
    if n > 1.0:
        f = f / n
    return np.clip(f, -1.0, 1.0)


def sample_goals():
    """与 verify_radius_generalize.py 完全一致：预采样 50 个跨障碍随机目标（train radius 0.03）。"""
    goals = []
    for s in range(N_EVAL):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(
            raw, obstacles=get_obstacles(True, MOTION, radius=TRAIN_RADIUS),
            success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
            collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST, hide_obs=True)
        obs, _ = e.reset(seed=10000 + s)
        goals.append(e._sample_cross_obstacle_goal())
        e.close()
    return goals


def run_apf(radius, goals):
    """经典 APF 在给定半径下跑 N_EVAL 个固定目标，返回与 verify 同口径的四格统计。"""
    n_succ = n_coll = n_safe = n_through = n_collide_fail = n_safe_fail = n_top = 0
    paths = []
    min_clears = []
    for s, goal in enumerate(goals):
        raw = gym.make("PandaReach-v3", reward_type="dense", max_episode_steps=100)
        raw.unwrapped.task.distance_threshold = SUCCESS_THRESHOLD
        e = MultiObstacleWrapper(
            raw, obstacles=get_obstacles(True, MOTION, radius=radius),
            success_threshold=SUCCESS_THRESHOLD, eval_mode=True,
            collision_penalty=COLLISION_PENALTY, step_cost=STEP_COST, hide_obs=True)
        obs, _ = e.reset(seed=10000 + s)
        obs = e._force_goal(obs, goal)

        traj = [obs["observation"][:3].copy()]
        ep_coll = ep_top = False
        min_clear = 1e9
        for _ in range(100):
            ee = obs["observation"][:3]
            for oc in e.obstacles:
                dc = float(np.hypot(ee[0] - oc["x"], ee[1] - oc["y"]))
                if 0.0 <= ee[2] <= oc["height"]:
                    min_clear = min(min_clear, dc - oc["radius"])
                if dc < oc["radius"] and ee[2] > oc["height"]:
                    ep_top = True
            a = apf_action(ee, goal, e.obstacles)
            obs, r, term, trunc, info = e.step(a)
            obs = dict(obs)
            obs["desired_goal"] = goal.copy()
            traj.append(obs["observation"][:3].copy())
            if info.get("is_collision", False):
                ep_coll = True
            if trunc or np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD:
                break
        e.close()

        P = np.array(traj)
        paths.append(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
        min_clears.append(min_clear)
        reached = np.linalg.norm(obs["observation"][:3] - goal) < SUCCESS_THRESHOLD
        collided = bool(ep_coll)
        if reached:
            n_succ += 1
            if collided:
                n_through += 1
            else:
                n_safe += 1
        else:
            if collided:
                n_collide_fail += 1
            else:
                n_safe_fail += 1   # 卡局部极小 / 100 步没走完（安全未达）
        n_coll += int(collided)
        n_top += int(ep_top)
    N = len(goals)
    return dict(succ=n_succ / N, coll=n_coll / N, top=n_top / N,
                safe=n_safe / N, through=n_through / N,
                stuck=n_safe_fail / N,
                path=float(np.mean(paths)), clear=float(np.mean(min_clears)))


def tune():
    """只跑 train radius 0.03，打印各指标，用于调 K_REP/RHO0 直到 0 撞柱。"""
    print(f"[tune] radius={TRAIN_RADIUS}  K_ATT={K_ATT} K_REP={K_REP} RHO0={RHO0} "
          f"K_Z={K_Z}", flush=True)
    goals = sample_goals()
    r = run_apf(TRAIN_RADIUS, goals)
    print(f"  成功率={r['succ']*100:.1f}%  撞柱率={r['coll']*100:.1f}%  "
          f"安全成功={r['safe']*100:.1f}%  穿柱到达={r['through']*100:.1f}%  "
          f"卡住={r['stuck']*100:.1f}%", flush=True)
    print(f"  路径={r['path']:.3f}m  最小clearance={r['clear']:.4f}m", flush=True)


def main():
    goals = sample_goals()
    rows = []
    print("\n=== 经典 APF 基线 · 6 半径扫描 ===", flush=True)
    for radius in RADII:
        r = run_apf(radius, goals)
        rows.append([radius, r["coll"], r["succ"], r["safe"], r["through"],
                     r["stuck"], r["top"], r["path"], r["clear"]])
        print(f"r={radius:.3f}  撞柱率={r['coll']*100:5.1f}%  成功率={r['succ']*100:5.1f}%  "
              f"安全成功={r['safe']*100:5.1f}%  卡住={r['stuck']*100:5.1f}%  "
              f"路径={r['path']:.3f}m  clear={r['clear']:.4f}m", flush=True)

    with open("classic_apf_radius.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "radius",
                    "collision_mean", "success_mean", "safe_success_mean",
                    "through_mean", "stuck_mean", "over_top_mean",
                    "path_mean", "clearance_mean"])
        for row in rows:
            w.writerow(["classic_apf"] + [f"{x:.6f}" if isinstance(x, float) else x for x in row])

    plot_figure(rows)
    print(f"\n[已保存] classic_apf_radius.csv", flush=True)


def plot_figure(rows):
    """单独 2×2 面板：撞柱率 / 安全成功率 / 路径长度 / 最小 clearance vs 半径。
    rows 列序: [radius, coll, succ, safe, through, stuck, top, path, clear]。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Classic APF planner (non-RL) — size robustness (train r=0.03)",
                 fontsize=14)
    rad = [r[0] for r in rows]
    panels = [
        ("coll", "Collision rate", 1),
        ("safe", "Safe success rate (reach ∧ no collision)", 3),
        ("path", "Trajectory length (m)", 7),
        ("clear", "Min clearance (m)", 8),
    ]
    for panel_idx, (key, ylabel, col_idx) in enumerate(panels):
        ax = axes[panel_idx // 2][panel_idx % 2]
        ys = [r[col_idx] for r in rows]
        ax.plot(rad, ys, color="#2ca02c", lw=2.5, marker="o", ms=6, label="classic APF")
        ax.axvline(TRAIN_RADIUS, color="gray", ls=":", lw=1.5, alpha=0.8)
        ax.set_xlabel("Obstacle radius (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97+panel_idx)}) {ylabel}", fontsize=12)
        ax.grid(True, alpha=0.3)
        if key in ("coll", "safe"):
            ax.set_ylim(-0.03, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = "Paper_Figure_ClassicAPF.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[已保存] {out}", flush=True)


def scan():
    """在 train radius 0.03 上扫 K_REP，检验结果对斥力增益是否敏感。"""
    goals = sample_goals()
    print(f"\n=== K_REP 扫描 (radius={TRAIN_RADIUS}, K_ATT={K_ATT}, RHO0={RHO0}) ===", flush=True)
    print(f"{'K_REP':>6}  {'撞柱率':>8}  {'未达率':>8}  {'安全成功':>8}  {'路径':>7}  {'clear':>7}", flush=True)
    for krep in [0.01, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0]:
        global K_REP
        K_REP = krep
        r = run_apf(TRAIN_RADIUS, goals)
        print(f"{krep:6.1f}  {r['coll']*100:7.1f}%  {r['stuck']*100:7.1f}%  "
              f"{r['safe']*100:7.1f}%  {r['path']:6.3f}m  {r['clear']:6.4f}m", flush=True)


def control():
    """对照：关掉斥力（K_REP=0，忽略柱子）的直线吸引控制器在 train radius 0.03 的到达率。
    若到达率也 <100%，说明未达目标与柱子无关（运动学/步数限制）。"""
    global K_REP
    K_REP = 0.0
    goals = sample_goals()
    r = run_apf(TRAIN_RADIUS, goals)
    print(f"[control] K_REP=0 (pillars ignored): reach={r['succ']*100:.1f}%  collision={r['coll']*100:.1f}%  "
          f"path={r['path']:.3f}m", flush=True)


def plot():
    """从 classic_apf_radius.csv 重绘图（不重跑扫描）。"""
    rows = []
    with open("classic_apf_radius.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append([float(r["radius"]), float(r["collision_mean"]),
                         float(r["success_mean"]), float(r["safe_success_mean"]),
                         float(r["through_mean"]), float(r["stuck_mean"]),
                         float(r["over_top_mean"]), float(r["path_mean"]),
                         float(r["clearance_mean"])])
    plot_figure(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        tune()
    elif len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan()
    elif len(sys.argv) > 1 and sys.argv[1] == "control":
        control()
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        main()
