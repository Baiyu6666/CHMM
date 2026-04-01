# visualization/pybullet_renderer.py
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
import os, sys
import matplotlib.pyplot as plt
from pathlib import Path


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from visualization.io import save_figure


class PyBulletRenderer3D:
    def __init__(
        self,
        env,
        panda_base_pos=(-0.25, -0.55, 0.0),  # z   "      ",setup        
        panda_base_rpy=(0, 0, 1.57),

        table_size=(1.6, 1.2, 0.75),   # fallback box      
        animate_fps=120,

        sphere_radius=0.012,           #      
        world_scale=0.5,               #    env   ,   Panda reach
        world_offset=(0.0, 0.0, 0.0),  # env -> world    (x,y,z),z          

        cylinder_height=0.6,           #        
        mug_scale=2.,                 #          
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

        # English comment omitted during cleanup.
        self.table_top_z = None
        self.table_id = None
        self.panda_id = None
        self.ee_link = None

        self.arm_joints = []
        self.arm_lower = []
        self.arm_upper = []
        self.arm_ranges = []
        self.arm_rest = []

        # English comment omitted during cleanup.
        self.held_obj_id = None
        self.held_constraint_id = None
        self._last_q_arm = None  #          


    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _world(self, pos):
        """
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
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
    # English comment omitted during cleanup.
    # ==========================================================
    def _load_table(self):
        """
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
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

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
        pb = list(self.panda_base_pos)
        if self.table_top_z is None:
            raise RuntimeError("table_top_z is None, call _load_table() first.")
        if pb[2] < self.table_top_z - 1e-4:
            pb[2] = self.table_top_z
        pb[2] += 1e-3  #    z-fighting
        self.panda_base_pos = tuple(pb)

        self.panda_id = p.loadURDF(
            panda_urdf,
            basePosition=self.panda_base_pos,
            baseOrientation=p.getQuaternionFromEuler(self.panda_base_rpy),
            useFixedBase=True
        )

        # English comment omitted during cleanup.
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
            self.ee_link = 11  #    panda hand index

        # English comment omitted during cleanup.
        for j in range(num_joints):
            info = p.getJointInfo(self.panda_id, j)
            j_name = info[1].decode("utf-8")
            if "finger" in j_name:
                p.resetJointState(self.panda_id, j, 0.04)

    def _spawn_held_object_at_ee(self):
        """
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
        """
        # English comment omitted during cleanup.
        if self.held_constraint_id is not None:
            p.removeConstraint(self.held_constraint_id)
            self.held_constraint_id = None
        if self.held_obj_id is not None:
            p.removeBody(self.held_obj_id)
            self.held_obj_id = None

        # English comment omitted during cleanup.
        ls = p.getLinkState(self.panda_id, self.ee_link, computeForwardKinematics=True)
        ee_pos = ls[0]
        ee_orn = ls[1]

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
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
    # English comment omitted during cleanup.
    # English comment omitted during cleanup.
    # English comment omitted during cleanup.
    # ==========================================================
    def _load_obstacle(self):
        """
        English documentation omitted during cleanup.

        English documentation omitted during cleanup.
              English documentation omitted during cleanup.
        English documentation omitted during cleanup.
        """
        # English comment omitted during cleanup.
        if hasattr(self.env, "centerline"):
            self._load_sine_surface()
            return

        # English comment omitted during cleanup.
        cx, cy = self.env.obs_center_xy
        r = self.env.obs_radius

        # English comment omitted during cleanup.
        try:
            mug_urdf = "objects/mug.urdf"
            mug_pos = self._world([cx, cy, 0.10])  #         
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

        # English comment omitted during cleanup.
        cyl_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=r * self.world_scale,
            length=self.cylinder_height,
            rgbaColor=[0.7, 0.7, 0.7, 0.4],
        )
        cyl_pos = self._world([cx, cy, self.cylinder_height / 2.0])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cyl_vis,
            basePosition=cyl_pos.tolist(),
        )

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def _load_sine_surface(
            self,
            n_segments: int = 70,
            thickness_env: float = 0.04,
            height_world: float | None = None,
    ):
        """
        English documentation omitted during cleanup.

            English documentation omitted during cleanup.
            - y = centerline(x)
            English documentation omitted during cleanup.
            English documentation omitted during cleanup.

        English documentation omitted during cleanup.
        """
        if height_world is None:
            height_world = self.cylinder_height  #               

        # English comment omitted during cleanup.
        xs_candidates = []

        # English comment omitted during cleanup.
        if hasattr(self.env, "x_start_range"):
            try:
                xs_candidates.extend(list(self.env.x_start_range))
            except Exception:
                pass

        # English comment omitted during cleanup.
        if hasattr(self.env, "subgoal"):
            xs_candidates.append(float(self.env.subgoal[0]))
        if hasattr(self.env, "goal"):
            xs_candidates.append(float(self.env.goal[0]))

        if len(xs_candidates) == 0:
            # English comment omitted during cleanup.
            x_min_env, x_max_env = -1., 1.5
        else:
            x_min_env = min(xs_candidates) - 0.2
            x_max_env = max(xs_candidates) + 0.2

        # English comment omitted during cleanup.
        dx_env = (x_max_env - x_min_env) / float(n_segments)
        dx_world = dx_env * self.world_scale
        thickness_world = thickness_env * self.world_scale
        h_world = float(height_world)

        # English comment omitted during cleanup.
        half_extents = [
            dx_world / 2.0,
            thickness_world / 2.0,
            h_world / 3.0,
        ]

        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.7, 1.0, 0.35],  #       
        )

        # English comment omitted during cleanup.
        col_id = -1

        # English comment omitted during cleanup.
        for i in range(n_segments):
            x_env = x_min_env + (i + 0.5) * dx_env
            # English comment omitted during cleanup.
            try:
                y_env = float(self.env.centerline(x_env))
            except Exception:
                # English comment omitted during cleanup.
                continue

            # English comment omitted during cleanup.
            pos_env = [x_env, y_env, h_world / 2.0]
            pos_w = self._world(pos_env)  #      z        

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos_w.tolist(),
            )

        print(
            f"[Renderer] Loaded sine corridor surface: "
            f"xin[{x_min_env:.2f},{x_max_env:.2f}], segments={n_segments}"
        )

    # ==========================================================
    # English comment omitted during cleanup.
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
    # English comment omitted during cleanup.
    # ==========================================================
    def _solve_ik(self, target_pos_w, target_orn_w=None):
        """
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
        """
        if target_orn_w is None:
            cur = p.getLinkState(self.panda_id, self.ee_link)
            target_orn_w = cur[1]

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
        self._load_table()

        # 2) Panda
        self._load_panda()

        self._last_q_arm = None

        # English comment omitted during cleanup.
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1.0 / self.animate_fps)

        # English comment omitted during cleanup.
        self._spawn_held_object_at_ee()

        # English comment omitted during cleanup.
        self._load_obstacle()

        # English comment omitted during cleanup.
        self._reset_camera()

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def animate_demo_with_ik(self, X, tau=None,
                             v_target=0.25,   #         (m/s)
                             min_dt=1/120,    #        
                             max_dt=0.12,     #        
                             debug_print=False):
        """
        X: (T,3) env coords
        English documentation omitted during cleanup.
        """
        X = np.asarray(X, dtype=float)
        T = len(X)
        if T < 2:
            return

        # English comment omitted during cleanup.
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

        # English comment omitted during cleanup.
        if tau is not None:
            for t in range(T):
                pos_w = self._world(X[t])
                c = (0.2, 0.4, 1, 0.9) if t <= tau else (1, 0.3, 0.3, 0.9)
                self._create_sphere(pos_w, color=c)

            # English comment omitted during cleanup.
            if hasattr(self.env, "subgoal_xy") and hasattr(self.env, "subgoal_z"):
                g1 = np.array([self.env.subgoal_xy[0], self.env.subgoal_xy[1], self.env.subgoal_z])
                self._create_sphere(self._world(g1), color=(1, 0.5, 0, 1))
            if hasattr(self.env, "goal_xy") and hasattr(self.env, "goal_z"):
                g2 = np.array([self.env.goal_xy[0], self.env.goal_xy[1], self.env.goal_z])
                self._create_sphere(self._world(g2), color=(0, 1, 0, 1))

        # English comment omitted during cleanup.
        # English comment omitted during cleanup.
        dropped = False

        for t in range(1, T):
            p0 = X[t - 1]
            p1 = X[t]
            d = float(np.linalg.norm(p1 - p0))
            if d < 1e-6:
                d = 1e-6

            # English comment omitted during cleanup.
            dt_segment = np.clip(d / v_target, min_dt, max_dt)
            # English comment omitted during cleanup.
            n_sub = max(2, int(dt_segment * self.animate_fps))

            # English comment omitted during cleanup.
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

            # English comment omitted during cleanup.
            if (tau is not None) and (not dropped) and (t >= tau):
                if self.held_constraint_id is not None:
                    p.removeConstraint(self.held_constraint_id)
                    self.held_constraint_id = None
                    dropped = True
                    if debug_print:
                        print(f"[IK] Drop object at t={t} (tau={tau})")

    # ==========================================================
    # English comment omitted during cleanup.
    # ==========================================================
    def render_demo(self, X, tau, g1, g2, delay=0.003):
        """
        English documentation omitted during cleanup.
        English documentation omitted during cleanup.
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
        English documentation omitted during cleanup.
        """
        for i, (X, tau) in enumerate(zip(demos, taus)):
            print(f"[PyBullet] Demo {i}, T={len(X)}, tau={tau}")

            # English comment omitted during cleanup.
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


def _compute_all_features_for_demo(env, X):
    """
    English documentation omitted during cleanup.
    English documentation omitted during cleanup.

    English documentation omitted during cleanup.
        F: ndarray, shape = (T, M)
        English documentation omitted during cleanup.
    """
    X = np.asarray(X, float)
    T = len(X)

    # English comment omitted during cleanup.
    if hasattr(env, "compute_all_features_matrix"):
        F = env.compute_all_features_matrix(X)
        F = np.asarray(F, float)
        M = F.shape[1]
        feat_names = [f"f{m}" for m in range(M)]
        return F, feat_names

    # English comment omitted during cleanup.
    elif hasattr(env, "compute_features_all"):
        d_raw, s_raw = env.compute_features_all(X)  # list/ndarray
        d_raw = np.asarray(d_raw, float)
        s_raw = np.asarray(s_raw, float)

        # English comment omitted during cleanup.
        if len(s_raw) < T:
            if len(s_raw) == 0:
                s_raw = np.zeros(T)
            else:
                s_raw = np.concatenate([s_raw, [s_raw[-1]]])

        F = np.stack([d_raw, s_raw], axis=1)  # (T, 2)
        feat_names = ["distance", "speed"]
        return F, feat_names

    else:
        # English comment omitted during cleanup.
        F = np.zeros((T, 1), float)
        feat_names = ["dummy"]
        return F, feat_names




def main():
    """
    English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
    """
    from envs.S5SphereInspect import S5SphereInspectEnv

    # English comment omitted during cleanup.
    mode = "sphere_inspect"

    if mode == "sphere_inspect":
        env = S5SphereInspectEnv()
        demos, cutpoints = env.generate_demos(n_demos=2)
        traj3d = demos[0]
        tau = int(cutpoints[0][0]) if cutpoints and len(cutpoints[0]) > 0 else 0
        g1 = getattr(env, "stage_end_markers", [None])[0]
        g2 = getattr(env, "stage_end_markers", [None, None])[-1]
    
        # ==========================================================
        # English comment omitted during cleanup.
        # ==========================================================
        X0 = demos[0]
        tau0 = int(taus[0]) if (taus is not None and len(taus) > 0) else None

        F, feat_names = _compute_all_features_for_demo(env, X0)  # (T, M)
        T0, M = F.shape
        t_axis = np.arange(T0)

        plt.figure(figsize=(10, 6))
        for m in range(M):
            plt.plot(t_axis, F[:, m], label=feat_names[m])

        if tau0 is not None:
            plt.axvline(tau0, color="red", linestyle="--", label="true tau")

        plt.xlabel("t")
        plt.ylabel("feature value")
        plt.title("Features over time (demo 0)")
        plt.legend(loc="best")
        plt.tight_layout()
        save_figure(
            plt.gcf(),
            Path(_PROJECT_ROOT) / "outputs" / "plots" / "PyBulletRenderer3D" / "features_over_time.png",
        )


    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # English comment omitted during cleanup.
    renderer = PyBulletRenderer3D(
        env,
        world_scale=0.5,
        world_offset=(0.0, 0.0, 0.0),
        table_size=(1.6, 1.2, 0.75),
        panda_base_pos=(-0.25, -0.55, 0.0),
    )
    renderer.setup_scene()

    print(f"Running IK debug animation in mode='{mode}' ...")
    renderer.play_all([traj3d], [tau], g1, g2,
                      v_target=0.20, min_dt=1/120, max_dt=0.10)

    print("Renderer debug done. Close window to exit.")
    while p.isConnected():
        time.sleep(0.1)


if __name__ == "__main__":
    main()
