import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np

# 双障碍场景配置：两柱略对角线错开（柱 A 偏上 y=0.045、柱 B 偏下 y=0.015），半径 0.03，
# 起点→目标直线仍必撞两柱；机械臂需「单侧绕行」或「翻顶」而非直线穿越。
# 几何依据（已数值核验）：
#   起点 ee ≈ (0.038, 0, 0.197)，目标采样 y∈[0.03,0.15]、z∈[0.08,0.13]（柱带中低部，低于柱顶）。
#   柱高 0.21；柱 A（x=-0.07,y=0.045）、柱 B（x=-0.19,y=0.015），半径 0.03。
#   起点→目标直线距柱 A≈0.027、距柱 B≈0.022，均 < 0.03 → 直线必撞两柱。
#   起点距柱 A≈0.117（不撞柱）；动态反相最近 2D≈0.095 > 半径和 0.06（不重叠）。
#   错开方向为「A 上 B 下」对角线，留出自然的 S/Z 绕行通道（而非同侧平移）。
#   目标 z 设在中低部（≈0.10）；翻顶(z>0.21)仍是较短路径且 z>0.21 时 APF/碰撞均不触发，
#   safety_only(APF,无height) 是否翻顶、ours(height惩罚) 是否低空绕行 → 由机制涌现决定。
FIXED_GOAL = np.array([-0.33, 0.06, 0.10], dtype=np.float32)
SAFE_MARGIN = 0.08

# 每个障碍: {x, y, radius, height, amplitude, period, phase, axis}
# axis: 'x' 或 'y'，动态时柱心沿该轴运动。
# motion: 'sine'（正弦可预测）| 'random_walk'（随机游走不可预测，C 方案）。
# 注意：柱按 x 从大到小（先遇到的在前）排列，便于按顺序描述几何。
def get_obstacles(dynamic=False, motion="random_walk", radius=0.03):
    # radius 覆盖：评估期扫半径（0.03 → 0.055）检验「尺寸鲁棒性」。
    # 注意：碰撞判定 / APF 势场 / 观测特征都读 oc["radius"]，改这里即全链路生效。
    obs = [
        dict(x=-0.07, y=0.045, radius=radius, height=0.21,
             amplitude=0.0, period=40, phase=0.0, axis="x",
             motion="static", walk_step=0.0),
        dict(x=-0.19, y=0.015, radius=radius, height=0.21,
             amplitude=0.0, period=60, phase=0.0, axis="x",
             motion="static", walk_step=0.0),
    ]
    if dynamic:
        obs[0]["amplitude"] = 0.015
        obs[1]["amplitude"] = 0.015
        obs[1]["phase"] = np.pi  # 反相（仅 sine 模式使用）
        for oc in obs:
            oc["motion"] = motion  # "sine" 或 "random_walk"
            oc["walk_step"] = 0.005  # 随机游走每步最大位移（与正弦峰值速度相当）
    return obs


OBSTACLES = get_obstacles(dynamic=False)

DEFAULT_SAFETY_WEIGHT = 1.5   # 3.0 -> 1.5：势场过重会压制"钻柱底"收益，导致策略瘫到桌面(z≈0)
DEFAULT_SMOOTHNESS_WEIGHT = 0.03  # 0.5 -> 0.03：rate 平滑会误罚绕行换向，过重时策略宁撞柱也不绕行
DEFAULT_SMOOTH_MODE = "rate"      # "jerk"(罚控制量突变) | "rate"(幅度-方向解耦)
DEFAULT_COLLISION_PENALTY = 0.0   # 优化型开关：>0 时撞柱不终止、每步扣该值（接触时长成正比）；=0 时保持「撞柱=终止」可行性语义
DEFAULT_STEP_COST = 0.0         # 时间惩罚：每步固定扣该值。>0 时「绕行花时间」变贵，与撞柱惩罚构成效率↔安全 trade-off，
                                #   最优解落在「贴边刚好不撞」（而非绕大圈或直线穿），APF 势场在此 regime 下才有区分度。
DEFAULT_SUCCESS_BONUS = 5.0      # 到达 goal 的额外正反馈（vanilla 设 0，只用纯 dense -d）
DEFAULT_HEIGHT_PENALTY = 2.0
DEFAULT_PROGRESS_WEIGHT = 1.0
CROSS_OBSTACLE_GOAL_PROB = 1.0   # 0.7 -> 1.0：训练 100% 跨障碍，删除"不躲柱"白送任务，避免稀释绕行信号


class MultiObstacleWrapper(gym.Wrapper):
    """
    多障碍避障包装器（泛化自 SafeObstacleWrapper）：
    - 支持 N 个障碍，每个障碍可静态或动态（沿 x/y 轴正弦往返）。
    - 观测追加 4N 维特征（每个障碍 [dx, dy, clearance, z_to_top]）。
    - 水平势场、越顶惩罚、几何碰撞判定、动作平滑。
    """

    def __init__(
        self,
        env,
        obstacles=None,
        success_threshold=0.03,
        smoothness_weight=DEFAULT_SMOOTHNESS_WEIGHT,
        safety_weight=DEFAULT_SAFETY_WEIGHT,
        success_bonus=DEFAULT_SUCCESS_BONUS,
        height_penalty=DEFAULT_HEIGHT_PENALTY,
        progress_weight=DEFAULT_PROGRESS_WEIGHT,
        collision_penalty=DEFAULT_COLLISION_PENALTY,
        step_cost=DEFAULT_STEP_COST,
        hide_obs=False,
        eval_mode=False,
        physical_obstacle=False,
        smooth_mode=DEFAULT_SMOOTH_MODE,
        safe_margin=SAFE_MARGIN,
        svr_potential=None,
        collision_inflation=0.0,
    ):
        super().__init__(env)
        self.eval_mode = eval_mode
        self.success_threshold = success_threshold
        self.smoothness_weight = smoothness_weight
        self.safety_weight = safety_weight
        self.height_penalty = height_penalty
        self.success_bonus = success_bonus
        self.progress_weight = progress_weight
        self.collision_penalty = collision_penalty  # 优化型开关：>0 撞柱每步扣分不终止；=0 撞柱即终止（旧可行性语义）
        self.step_cost = step_cost  # 时间惩罚：每步固定扣，绕行变贵 → 效率↔安全 trade-off
        self.hide_obs = hide_obs  # True=隐掉障碍观测特征（部分可观测），APF 势场成为唯一障碍感知通道
        self.physical_obstacle = physical_obstacle
        self.smooth_mode = smooth_mode  # "jerk"（罚控制量突变）| "rate"（幅度-方向解耦，只罚速率突变）
        self.safe_margin = safe_margin
        # 膨胀碰撞半径对照（仅影响「碰撞惩罚项」的触发边界，不影响 info["is_collision"] 与终止判定）：
        # 默认 0.0 → 与原行为完全一致；>0 时每步在 dc < radius + collision_inflation 内扣 collision_penalty。
        self.collision_inflation = collision_inflation
        # 判别式势场（P2 消融）：非 None 时替换解析 APF 水平势场（读 EE 绝对坐标 (x,y)）
        self.svr_potential = svr_potential
        # 关键：让基础环境的成功阈值与本包装器一致，否则基础环境在 0.05 处就提前终止
        self.unwrapped.task.distance_threshold = success_threshold

        # 障碍配置深拷贝，保存初始中心（动态时作为运动中心）
        self.obstacles = []
        for o in (obstacles if obstacles is not None else OBSTACLES):
            oc = dict(o)
            oc["x0"] = oc["x"]
            oc["y0"] = oc["y"]
            self.obstacles.append(oc)
        self.n_obs = len(self.obstacles)
        self._pillar_height = max((oc["height"] for oc in self.obstacles), default=0.21)

        self._client = None
        self.obstacle_ids = []
        self.step_count = 0
        self.prev_action = None
        self.prev_dist_to_target = None

        # 观测追加 4N 维障碍特征（hide_obs=True 时隐掉，观测保持原始维度 → 部分可观测）
        old = env.observation_space["observation"]
        if not hide_obs:
            low = np.concatenate([old.low, np.full(4 * self.n_obs, -10.0, dtype=np.float32)])
            high = np.concatenate([old.high, np.full(4 * self.n_obs, 10.0, dtype=np.float32)])
            self.observation_space = spaces.Dict(
                {
                    **env.observation_space.spaces,
                    "observation": spaces.Box(low=low, high=high, dtype=np.float32),
                }
            )
        else:
            self.observation_space = env.observation_space

    # ---------------- 基础设施 ----------------
    def _get_client(self):
        if self._client is None:
            self._client = self.unwrapped.sim.physics_client
        return self._client

    def _spawn_obstacles_once(self):
        if not self.obstacle_ids:
            client = self._get_client()
            for oc in self.obstacles:
                vis = client.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=oc["radius"],
                    length=oc["height"],
                    rgbaColor=[1.0, 0.0, 0.0, 0.8],
                )
                col = -1
                if self.physical_obstacle:
                    col = client.createCollisionShape(
                        shapeType=p.GEOM_CYLINDER,
                        radius=oc["radius"],
                        height=oc["height"],
                    )
                bid = client.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[oc["x"], oc["y"], oc["height"] / 2],
                )
                self.obstacle_ids.append(bid)

    def _update_obstacles(self):
        """动态障碍：按 motion 模式更新柱心位置（正弦可预测 / 随机游走不可预测）。"""
        client = self._get_client()
        t = self.step_count
        rng = self.unwrapped.task.np_random
        for oc, bid in zip(self.obstacles, self.obstacle_ids):
            if oc["amplitude"] <= 0:
                continue
            if oc.get("motion") == "random_walk":
                # 随机游走：每步在 [-walk_step, walk_step] 均匀采样，边界 [x0±amplitude] 反弹。
                # 不可预测 → 无法"等时机穿越"，只能靠即时避障（APF 势场）反应。
                step = oc.get("walk_step", 0.005)
                amp = oc["amplitude"]
                if oc["axis"] == "x":
                    lo, hi = oc["x0"] - amp, oc["x0"] + amp
                    new = oc["x"] + rng.uniform(-step, step)
                    if new > hi:
                        new = 2 * hi - new  # 边界反弹
                    elif new < lo:
                        new = 2 * lo - new
                    oc["x"] = new
                else:
                    lo, hi = oc["y0"] - amp, oc["y0"] + amp
                    new = oc["y"] + rng.uniform(-step, step)
                    if new > hi:
                        new = 2 * hi - new
                    elif new < lo:
                        new = 2 * lo - new
                    oc["y"] = new
            else:  # sine（可预测，旧行为）
                phase = oc["phase"]
                if oc["axis"] == "x":
                    oc["x"] = oc["x0"] + oc["amplitude"] * np.sin(2 * np.pi * t / oc["period"] + phase)
                    oc["y"] = oc["y0"]
                else:
                    oc["y"] = oc["y0"] + oc["amplitude"] * np.sin(2 * np.pi * t / oc["period"] + phase)
                    oc["x"] = oc["x0"]
            client.resetBasePositionAndOrientation(
                bid,
                [oc["x"], oc["y"], oc["height"] / 2.0],
                [0.0, 0.0, 0.0, 1.0],
            )
        self.step_count += 1

    def _obstacle_features(self, ee_pos):
        feats = []
        for oc in self.obstacles:
            dx = oc["x"] - float(ee_pos[0])
            dy = oc["y"] - float(ee_pos[1])
            dist = float(np.hypot(ee_pos[0] - oc["x"], ee_pos[1] - oc["y"]))
            clearance = dist - oc["radius"]
            z_to_top = oc["height"] - float(ee_pos[2])
            feats.extend([dx, dy, clearance, z_to_top])
        return np.array(feats, dtype=np.float32)

    def _augment_obs(self, obs):
        if self.hide_obs:
            return obs  # 隐掉障碍观测：不追加任何障碍特征
        obs = dict(obs)
        feat = self._obstacle_features(obs["observation"][0:3])
        obs["observation"] = np.concatenate(
            [np.asarray(obs["observation"], dtype=np.float32), feat]
        ).astype(np.float32)
        return obs

    def _sample_cross_obstacle_goal(self):
        rng = self.unwrapped.task.np_random
        for _ in range(50):
            goal = np.array(
                [
                    rng.uniform(-0.38, -0.30),
                    rng.uniform(0.03, 0.15),
                    rng.uniform(0.08, 0.13),
                ],
                dtype=np.float32,
            )
            # 目标不能落在任一柱内
            ok = all(np.hypot(goal[0] - oc["x0"], goal[1] - oc["y0"]) > oc["radius"] + 0.03
                     for oc in self.obstacles)
            if ok:
                return goal
        return FIXED_GOAL.copy()

    def _force_goal(self, obs, goal):
        goal = np.asarray(goal, dtype=np.float32)
        self.unwrapped.task.goal = goal.copy()
        self.unwrapped.sim.set_base_pose("target", goal, np.array([0.0, 0.0, 0.0, 1.0]))
        obs = dict(obs)
        obs["desired_goal"] = goal.copy()
        return obs

    # ---------------- 环境接口 ----------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        for oc in self.obstacles:
            oc["x"] = oc["x0"]
            oc["y"] = oc["y0"]
        self._spawn_obstacles_once()
        self.prev_action = None
        self.prev_dist_to_target = None
        self._collision_steps = 0

        if not self.eval_mode:
            rng = self.unwrapped.task.np_random
            if rng.random() < CROSS_OBSTACLE_GOAL_PROB:
                obs = self._force_goal(obs, self._sample_cross_obstacle_goal())

        return self._augment_obs(obs), info

    def _height_cost(self, ee_pos):
        """高度惩罚（单边）：只罚翻顶（ee 高于任一柱顶）。

        语义：阻止机械臂「翻越」障碍物。relu(z - height)：
          - 低于柱顶：零惩罚，允许在柱顶以下自由低空绕行。
          - 高于柱顶：随超出量增大（线性 + 二次放大）。
        不惩罚钻底（z 过低）——那由碰撞带(0≤z≤height)和 APF 水平势场负责，属碰撞问题而非高度问题。
        归一化尺度 0.04（每超出 4cm 记 1 单位）。
        """
        z = float(ee_pos[2])
        over = max(0.0, max(z - oc["height"] for oc in self.obstacles))
        return (over / 0.04) + (over / 0.04) ** 2

    def step(self, action):
        self._update_obstacles()
        obs, reward, terminated, truncated, info = self.env.step(action)
        target_pos = obs["desired_goal"]
        ee_pos = obs["observation"][0:3]

        dist_to_target = float(np.linalg.norm(ee_pos - target_pos))
        is_success = dist_to_target < self.success_threshold

        # 1) 动作平滑
        #    - "jerk"：罚相邻步 action 向量突变（幅度+方向混在一起）
        #    - "rate"：幅度-方向解耦，只罚速率突变 (‖a_t‖ - ‖a_{t-1}‖)²，允许平滑绕行
        smooth_cost = 0.0
        if self.prev_action is not None:
            a = np.asarray(action, dtype=np.float32)
            if self.smooth_mode == "rate":
                n_cur = float(np.linalg.norm(a))
                n_prev = float(np.linalg.norm(self.prev_action))
                smooth_cost = (n_cur - n_prev) ** 2
            else:  # "jerk"
                smooth_cost = float(np.sum((a - self.prev_action) ** 2))

        # 2) 水平势场：解析 APF（默认）或判别式 SVR 势场（P2 消融，svr_potential 非 None 时）。
        #    解析势场 clearance=dc−radius，radius 显式 → 评估期随真实半径自动外扩；
        #    判别式势场只读 EE 绝对水平坐标 (x,y)，radius/柱位置烤进权重 → OOD 半径下边界不随半径外扩。
        if self.svr_potential is not None:
            safety_cost = 0.0
            if 0.0 <= ee_pos[2] <= self._pillar_height:
                safety_cost = float(self.svr_potential.predict(
                    np.asarray([[ee_pos[0], ee_pos[1]]], dtype=np.float64))[0])
                safety_cost = max(0.0, safety_cost)
        else:
            # 解析势场（对所有障碍求和）：只在 ee 处于柱体高度带内（0≤z≤height）才排斥，
            # 防止"柱体高度内的水平穿越"（会撞柱）。翻顶（z>height）不撞柱，势场不介入。
            safety_cost = 0.0
            for oc in self.obstacles:
                dc = float(np.hypot(ee_pos[0] - oc["x"], ee_pos[1] - oc["y"]))
                clearance = dc - oc["radius"]
                if 0.0 <= ee_pos[2] <= oc["height"] and clearance < self.safe_margin:
                    depth = (self.safe_margin - max(clearance, 0.0)) / self.safe_margin
                    safety_cost += depth ** 2

        # 2b) 高度惩罚（单边翻顶，见 _height_cost）
        height_cost = self._height_cost(ee_pos)

        # 3) 碰撞判定：ee 在柱体高度带内（0≤z≤height）水平进入柱体 → 撞柱。
        #    翻顶（z>height）水平穿越不算撞柱（记录为越顶率），可合法绕行。
        #    语义由 collision_penalty 决定：
        #      =0（可行性）→ 撞柱即终止，靠「终止 + 拿不到 success_bonus」提供信号；
        #      >0（优化型）→ 撞柱不终止，每步扣固定值 collision_penalty（线性 per-step），
        #                     逼策略在「绕行代价」与「撞柱代价」之间找平衡，APF 势场在此 regime 下才有区分度。
        is_collision = False
        is_penalized = False  # 与 is_collision 相同，除非启用 collision_inflation（膨胀半径对照）
        for oc in self.obstacles:
            dc = float(np.hypot(ee_pos[0] - oc["x"], ee_pos[1] - oc["y"]))
            in_band = 0.0 <= ee_pos[2] <= oc["height"]
            if dc < oc["radius"] and in_band:
                is_collision = True
            if dc < oc["radius"] + self.collision_inflation and in_band:
                is_penalized = True
        if is_collision and self.collision_penalty <= 0.0:
            terminated = True

        success_bonus = self.success_bonus if is_success else 0.0

        # 纯「朝 goal」进度（奖励整形，通用项）：只奖励接近 goal，不惩罚远离。
        if self.prev_dist_to_target is None:
            self.prev_dist_to_target = dist_to_target
        progress_reward = max(0.0, (self.prev_dist_to_target - dist_to_target)) * self.progress_weight
        self.prev_dist_to_target = dist_to_target

        total_reward = (
            float(reward)
            + success_bonus
            + progress_reward
            - self.smoothness_weight * smooth_cost
            - self.safety_weight * safety_cost
            - self.height_penalty * height_cost
            - self.collision_penalty * float(is_penalized)  # 线性 per-step 撞柱惩罚（collision_penalty=0 时无此项）
            - self.step_cost  # 时间惩罚：每步固定扣，绕行(步数多)变贵 → 效率↔安全 trade-off
        )

        info = dict(info)
        info["is_collision"] = is_collision
        info["total_reward"] = total_reward
        # 累计撞柱步数（优化型语义下用于评估「撞柱频率/时长」）
        self._collision_steps = getattr(self, "_collision_steps", 0) + int(is_collision)
        info["collision_steps"] = self._collision_steps
        self.prev_action = np.asarray(action, dtype=np.float32).copy()

        return self._augment_obs(obs), total_reward, terminated, truncated, info
