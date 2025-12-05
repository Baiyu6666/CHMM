# render/pybullet_renderer.py
# ------------------------------------------------------------
# PyBullet visualization:
#   - textured table (URDF if available, else box + texture)
#   - Franka Panda (fixed on tabletop)
#   - mug obstacle + transparent constraint cylinder
#   - held cube in gripper (dropped at subgoal)
#   - trajectory playback with IK + interpolation
# ------------------------------------------------------------

import time
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletRenderer3D:
    def __init__(
        self,
        env,
        panda_base_pos=(-0.25, -0.55, 0.0),  # z 视为“相对桌面高度”，setup 里会挪到桌面上
        panda_base_rpy=(0, 0, 1.57),

        table_size=(1.6, 1.2, 0.75),   # fallback box 桌子的尺寸
        animate_fps=120,

        sphere_radius=0.012,           # 轨迹点半径
        world_scale=0.5,               # 缩小 env 尺度，方便 Panda reach
        world_offset=(0.0, 0.0, 0.0),  # env -> world 的平移（x,y,z），z 会再加到桌面高度上

        cylinder_height=0.6,           # 透明圆柱的高度
        mug_scale=2.,                 # 桌上杯子的放大倍数
    ):
        self.env = env
        self.panda_base_pos = panda_base_pos
        self.panda_base_rpy = panda_base_rpy

        self.table_size = table_size
        self.animate_fps = animate_fps
        self.sphere_radius = sphere_radius

        self.world_scale = float(world_scale)
        self.world_offset = np.array(world_offset, dtype=float)

        self.cylinder_height = float(cylinder_height)
        self.mug_scale = float(mug_scale)

        # 这些会在 setup_scene 里填充
        self.table_top_z = None
        self.table_id = None
        self.panda_id = None
        self.ee_link = None

        self.arm_joints = []
        self.arm_lower = []
        self.arm_upper = []
        self.arm_ranges = []
        self.arm_rest = []

        # 抓在手里的物体
        self.held_obj_id = None
        self.held_constraint_id = None
        self._last_q_arm = None  # 保存上一帧的关节角


    # ==========================================================
    # 坐标变换：env -> world（桌面坐标系）
    # ==========================================================
    def _world(self, pos):
        """
        pos: (2,) 或 (3,) in env coordinates.
        返回: (3,) in PyBullet world coordinates, z 自动加到桌面高度之上。
        """
        pos = np.asarray(pos, dtype=float)
        if pos.shape[0] == 2:
            pos = np.array([pos[0], pos[1], 0.0], dtype=float)

        w = self.world_scale * pos + self.world_offset
        if self.table_top_z is not None:
            w = w.copy()
            w[2] += self.table_top_z
        return w

    def _create_sphere(self, pos_w, color=(1, 0, 0, 1)):
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.sphere_radius,
            rgbaColor=color
        )
        body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=list(pos_w)
        )
        return body

    # ==========================================================
    # 桌子
    # ==========================================================
    def _load_table(self):
        """
        1) 优先用 pybullet_data/table/table.urdf（带木纹）
        2) 否则用 box + 贴 texture（checker/wood）
        设置: self.table_id, self.table_top_z
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # --- 1) URDF table ---
        try:
            table_urdf = "table/table.urdf"
            self.table_id = p.loadURDF(
                table_urdf,
                basePosition=[0, 0, 0],
                useFixedBase=True
            )
            aabb_min, aabb_max = p.getAABB(self.table_id, -1)
            self.table_top_z = aabb_max[2]
            print("[Renderer] Loaded URDF table:", table_urdf)
            return
        except Exception as e:
            print("[PyBulletRenderer3D] URDF table load failed, fallback to box:", e)

        # --- 2) fallback: box + texture ---
        L, W, H = self.table_size
        self.table_top_z = H

        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L / 2, W / 2, H / 2]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L / 2, W / 2, H / 2],
            rgbaColor=[0.9, 0.9, 0.9, 1.0]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0, 0, H / 2]
        )

        # 尝试加载木纹 / 棋盘纹理
        tex_id = None
        for tex_name in ["wood.png", "checker_grid.jpg", "checker_blue.png"]:
            try:
                tex_id = p.loadTexture(tex_name)
                print("[Renderer] Table texture loaded:", tex_name)
                break
            except Exception:
                continue

        if tex_id is not None:
            p.changeVisualShape(self.table_id, -1, textureUniqueId=tex_id)
        else:
            print("[Renderer] No table texture found, keep plain color.")

    # ==========================================================
    # Panda + joints + held cube
    # ==========================================================
    def _load_panda(self):
        panda_urdf = pybullet_data.getDataPath() + "/franka_panda/panda.urdf"

        # 让 base 落在桌面上方一点
        pb = list(self.panda_base_pos)
        if self.table_top_z is None:
            raise RuntimeError("table_top_z is None, call _load_table() first.")
        if pb[2] < self.table_top_z - 1e-4:
            pb[2] = self.table_top_z
        pb[2] += 1e-3  # 避免 z-fighting
        self.panda_base_pos = tuple(pb)

        self.panda_id = p.loadURDF(
            panda_urdf,
            basePosition=self.panda_base_pos,
            baseOrientation=p.getQuaternionFromEuler(self.panda_base_rpy),
            useFixedBase=True
        )

        # --- 收集 arm joints & ee link ---
        self.arm_joints = []
        self.arm_lower = []
        self.arm_upper = []
        self.arm_ranges = []
        self.arm_rest = []
        self.ee_link = None

        num_joints = p.getNumJoints(self.panda_id)
        for j in range(num_joints):
            info = p.getJointInfo(self.panda_id, j)
            j_name = info[1].decode("utf-8")
            j_type = info[2]
            link_name = info[12].decode("utf-8")

            if j_type == p.JOINT_REVOLUTE and j_name.startswith("panda_joint"):
                self.arm_joints.append(j)
                self.arm_lower.append(info[8])
                self.arm_upper.append(info[9])
                self.arm_ranges.append(info[9] - info[8])
                self.arm_rest.append(p.getJointState(self.panda_id, j)[0])

            if ("panda_hand" in link_name) or ("panda_grasptarget" in link_name):
                self.ee_link = j

        if self.ee_link is None:
            self.ee_link = 11  # 常见 panda hand index

        # 打开手指一点
        for j in range(num_joints):
            info = p.getJointInfo(self.panda_id, j)
            j_name = info[1].decode("utf-8")
            if "finger" in j_name:
                p.resetJointState(self.panda_id, j, 0.04)

    def _spawn_held_object_at_ee(self):
        """
        在当前 EE 位置生成一个长方体，并用 JOINT_FIXED 绑定。
        注意：这里用的是 world 坐标，不再经过 _world()，避免 scale 偏移。
        """
        # 删掉旧的
        if self.held_constraint_id is not None:
            p.removeConstraint(self.held_constraint_id)
            self.held_constraint_id = None
        if self.held_obj_id is not None:
            p.removeBody(self.held_obj_id)
            self.held_obj_id = None

        # 当前末端位姿（world）
        ls = p.getLinkState(self.panda_id, self.ee_link, computeForwardKinematics=True)
        ee_pos = ls[0]
        ee_orn = ls[1]

        # 方块大小（看得比较清楚）
        half_extents = [0.02, 0.02, 0.03]  # 8cm x 8cm x 12cm

        col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.6, 0.9, 1.0]
        )
        self.held_obj_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=ee_pos,
            baseOrientation=ee_orn
        )

        # 固定约束：直接用 0 offset，让方块中心对齐 hand link frame
        self.held_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.panda_id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=self.held_obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

    # ==========================================================
    # 障碍物: 杯子 + 透明圆柱
    # ==========================================================
    def _load_obstacle(self):
        cx, cy = self.env.obs_center_xy
        r = self.env.obs_radius

        # 杯子：用杯子的 URDF，放在圆柱中心附近
        try:
            mug_urdf = "objects/mug.urdf"
            mug_pos = self._world([cx, cy, 0.10])  # 杯子底大概在桌上
            p.loadURDF(
                mug_urdf,
                basePosition=mug_pos.tolist(),
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=self.mug_scale,
                useFixedBase=True
            )
            print("[Renderer] Loaded mug:", mug_urdf)
        except Exception as e:
            print("[Renderer] Mug load failed:", e)

        # 透明约束圆柱
        cyl_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=r * self.world_scale,
            length=self.cylinder_height,
            rgbaColor=[0.7, 0.7, 0.7, 0.4]
        )
        cyl_pos = self._world([cx, cy, self.cylinder_height / 2.0])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cyl_vis,
            basePosition=cyl_pos.tolist()
        )

    # ==========================================================
    # 摄像机
    # ==========================================================
    def _reset_camera(self):
        target = [0.2, 0.0, self.table_top_z]
        p.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=target
        )

    # ==========================================================
    # IK 求解
    # ==========================================================
    def _solve_ik(self, target_pos_w, target_orn_w=None):
        """
        连续 IK：
        - restPoses 优先用上一帧的关节角 self._last_q_arm
        - 如果还没有，就用初始的 self.arm_rest
        """
        if target_orn_w is None:
            cur = p.getLinkState(self.panda_id, self.ee_link)
            target_orn_w = cur[1]

        # 如果有上一帧的解，就用它当 rest pose，避免 IK 突然跳解
        if self._last_q_arm is not None:
            rest = list(self._last_q_arm)
        else:
            rest = list(self.arm_rest)

        q_full = p.calculateInverseKinematics(
            bodyUniqueId=self.panda_id,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=target_pos_w,
            targetOrientation=target_orn_w,
            lowerLimits=self.arm_lower,
            upperLimits=self.arm_upper,
            jointRanges=self.arm_ranges,
            restPoses=rest,
            maxNumIterations=80,
            residualThreshold=1e-4
        )

        q_arm = q_full[:len(self.arm_joints)]

        # 记住这一次的解，下一帧用作 rest pose
        self._last_q_arm = list(q_arm)
        return q_arm

    # ==========================================================
    # Scene setup
    # ==========================================================
    def setup_scene(self):
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.animate_fps)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")

        # 1) 桌子
        self._load_table()

        # 2) Panda
        self._load_panda()

        self._last_q_arm = None

        # 3) 先走几帧，让一切 settle 一下
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1.0 / self.animate_fps)

        # 4) 在当前 EE 位置生成一个方块并抓住
        self._spawn_held_object_at_ee()

        # 5) 杯子 + 透明圆柱
        self._load_obstacle()

        # 6) 设置摄像机
        self._reset_camera()

    # ==========================================================
    # IK 动画：自然速度 + 插值 + tau 处丢物体
    # ==========================================================
    def animate_demo_with_ik(self, X, tau=None,
                             v_target=0.25,   # 期望末端线速度 (m/s)
                             min_dt=1/120,    # 每小段最少时间
                             max_dt=0.12,     # 每小段最多时间
                             debug_print=False):
        """
        X: (T,3) env coords
        tau: cutpoint index (用于掉落与着色)
        """
        X = np.asarray(X, dtype=float)
        T = len(X)
        if T < 2:
            return

        # --- 0) warm-up: 把 EE 平滑对齐到 X[0] ---
        first_pos_w = self._world(X[0])
        cur_ls = p.getLinkState(self.panda_id, self.ee_link)
        cur_pos = np.array(cur_ls[0])
        cur_orn = cur_ls[1]

        steps_warmup = 30
        for k in range(steps_warmup):
            alpha = (k + 1) / steps_warmup
            pos_interp = (1 - alpha) * cur_pos + alpha * first_pos_w
            q_arm = self._solve_ik(pos_interp, target_orn_w=cur_orn)
            for jid, q in zip(self.arm_joints, q_arm):
                p.resetJointState(self.panda_id, jid, q)
            p.stepSimulation()
            time.sleep(1.0 / self.animate_fps)

        # --- 1) 可选：画点云（不再 stepSimulation，避免再次“瞬移”） ---
        if tau is not None:
            for t in range(T):
                pos_w = self._world(X[t])
                c = (0.2, 0.4, 1, 0.9) if t <= tau else (1, 0.3, 0.3, 0.9)
                self._create_sphere(pos_w, color=c)

            # g1/g2 如果你想画，可以从 env 拼
            if hasattr(self.env, "subgoal_xy") and hasattr(self.env, "subgoal_z"):
                g1 = np.array([self.env.subgoal_xy[0], self.env.subgoal_xy[1], self.env.subgoal_z])
                self._create_sphere(self._world(g1), color=(1, 0.5, 0, 1))
            if hasattr(self.env, "goal_xy") and hasattr(self.env, "goal_z"):
                g2 = np.array([self.env.goal_xy[0], self.env.goal_xy[1], self.env.goal_z])
                self._create_sphere(self._world(g2), color=(0, 1, 0, 1))

        # --- 2) 沿轨迹播放（线性插值 + v_target 控制速度） ---
        # tau 用于 drop 物体
        dropped = False

        for t in range(1, T):
            p0 = X[t - 1]
            p1 = X[t]
            d = float(np.linalg.norm(p1 - p0))
            if d < 1e-6:
                d = 1e-6

            # 理论上这一段应该花的时间
            dt_segment = np.clip(d / v_target, min_dt, max_dt)
            # 对应多少帧（向上取整）
            n_sub = max(2, int(dt_segment * self.animate_fps))

            # 线性插值
            for k in range(n_sub):
                alpha = (k + 1) / n_sub
                pos_env = (1 - alpha) * p0 + alpha * p1
                pos_w = self._world(pos_env)
                q_arm = self._solve_ik(pos_w)

                for jid, q in zip(self.arm_joints, q_arm):
                    p.resetJointState(self.panda_id, jid, q)

                p.stepSimulation()
                time.sleep(1.0 / self.animate_fps)

            if debug_print:
                print(f"[IK] t={t}/{T}, d={d:.3f}, dt_seg={dt_segment:.4f}, n_sub={n_sub}")

            # 在 tau 处丢物体（解除约束，让方块掉到桌面/杯子边上）
            if (tau is not None) and (not dropped) and (t >= tau):
                if self.held_constraint_id is not None:
                    p.removeConstraint(self.held_constraint_id)
                    self.held_constraint_id = None
                    dropped = True
                    if debug_print:
                        print(f"[IK] Drop object at t={t} (tau={tau})")

    # ==========================================================
    # 渲染接口（给 main.py 用）
    # ==========================================================
    def render_demo(self, X, tau, g1, g2, delay=0.003):
        """
        如果你只是想画静态点云，可以用这个。
        IK 动画请用 play_all（内部调 animate_demo_with_ik）。
        """
        X = np.asarray(X, dtype=float)
        T = len(X)

        # goals
        self._create_sphere(self._world(g1), color=(1, 0.5, 0, 1))   # g1 orange
        self._create_sphere(self._world(g2), color=(0, 1, 0, 1))     # g2 green

        # cutpoint
        if 0 <= tau < T:
            self._create_sphere(self._world(X[tau]), color=(1, 1, 0, 1))

        for t in range(T):
            pos_w = self._world(X[t])
            c = (0.2, 0.4, 1, 0.9) if t <= tau else (1, 0.3, 0.3, 0.9)
            self._create_sphere(pos_w, color=c)
            p.stepSimulation()
            time.sleep(delay)

    def play_all(self, demos, taus, g1, g2,
                 v_target=0.25, min_dt=1/120, max_dt=0.10):
        """
        demos: list of (T_i,3) arrays in env coords
        taus:  list of int cutpoints
        g1, g2: np.array(3,) learned goals（目前只用于 log，可不用）
        """
        for i, (X, tau) in enumerate(zip(demos, taus)):
            print(f"[PyBullet] Demo {i}, T={len(X)}, tau={tau}")

            # 每条 demo 重新在手上生成一个方块
            self._spawn_held_object_at_ee()

            self.animate_demo_with_ik(
                X,
                tau=tau,
                v_target=v_target,
                min_dt=min_dt,
                max_dt=max_dt,
                debug_print=False
            )

            time.sleep(0.8)


# ==========================================================
# Debug main（可选）：单独测试 Renderer
# ==========================================================
def main():
    class _FakeEnv:
        obs_center_xy = (-0.5, 0.0)
        obs_radius = 0.3
        subgoal_xy = (0.5, 0.0)
        goal_xy = (0.0, 0.3)
        subgoal_z = 0.3
        goal_z = 0.5

    env = _FakeEnv()
    renderer = PyBulletRenderer3D(
        env,
        world_scale=0.5,
        world_offset=(0.0, 0.0, 0.0),
        table_size=(1.6, 1.2, 0.75),
        panda_base_pos=(-0.25, -0.55, 0.0),
    )
    renderer.setup_scene()

    # dummy demo: arc around obstacle then go to goal
    T = 80
    theta = np.linspace(-np.pi / 2, np.pi / 2, T // 2)
    arc = np.stack([
        env.obs_center_xy[0] + (env.obs_radius + 0.25) * np.cos(theta),
        env.obs_center_xy[1] + (env.obs_radius + 0.25) * np.sin(theta),
        np.linspace(0.25, 0.35, len(theta))
    ], axis=1)
    tail = np.stack([
        np.linspace(arc[-1, 0], env.goal_xy[0], T - len(theta)),
        np.linspace(arc[-1, 1], env.goal_xy[1], T - len(theta)),
        np.linspace(arc[-1, 2], env.goal_z, T - len(theta))
    ], axis=1)
    X = np.concatenate([arc, tail], axis=0)

    tau = len(arc)
    g1 = np.array([env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z])
    g2 = np.array([env.goal_xy[0], env.goal_xy[1], env.goal_z])

    print("Running IK debug animation...")
    # 注意：这里不要再单独调用 render_demo，只用 play_all
    renderer.play_all([X], [tau], g1, g2,
                      v_target=0.20, min_dt=1/120, max_dt=0.10)

    print("Renderer debug done. Close window to exit.")
    while p.isConnected():
        time.sleep(0.1)


if __name__ == "__main__":
    main()
