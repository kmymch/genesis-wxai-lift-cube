import math
from functools import partial
from importlib.resources import files
from typing import Literal

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions
from genesis.vis.camera import Camera
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_trans_quat,
)

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False

# WidowX AI follower arm, shipped with trossen_arm_mujoco
WXAI_MJCF = str(files("trossen_arm_mujoco") / "assets" / "wxai" / "wxai_follower.xml")


class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg  # read by rsl-rl's runner
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)
        self.cube_size = env_cfg["box_size"]
        self.cube_z_center = self.cube_size[2] / 2.0

        # camera config
        self.wrist_cam_res = env_cfg["wrist_cam_resolution"]

        # == determine which environments to render ==
        # Genesis places environments in a ceil(sqrt(n_envs)) grid centered at the origin.
        # Render the 10 environments closest to the origin so the viewer shows them.
        grid_size = int(math.ceil(math.sqrt(self.num_envs)))
        if self.num_envs <= 10:
            rendered_envs_idx = list(range(self.num_envs))
        else:
            origin_env_idx = (grid_size // 2) * grid_size + grid_size // 2
            rendered_envs_idx = list(range(origin_env_idx - 5, origin_env_idx + 5))

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt),
            # two rigid substeps per control step
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt / 2,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=rendered_envs_idx,
                split_envs=True,
                ambient_light=(0.8, 0.8, 0.8),
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(2.5, -1.0, 2.5),
                camera_lookat=(0.5, -0.3, 0.1),
                camera_fov=55,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == add ground ==
        self.scene.add_entity(gs.morphs.Plane())

        # == add robot ==
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg.get("box_fixed", True),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.85, 0.1, 0.1),  # Red, to stand out from the blue floor
                ),
            ),
        )

        # == finger tip marker (hidden from the policy cameras in BC) ==
        if self.env_cfg.get("show_visual_helpers", True):
            self.finger_tip_vis = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.015,
                    fixed=True,
                    batch_fixed_verts=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.0, 0.0, 1.0),
                    ),
                ),
            )

        # == visualization camera (debug only, uses scene camera API) ==
        if self.env_cfg.get("visualize_camera", False):
            self.vis_cam = self.scene.add_camera(
                res=(1280, 960),
                pos=(0.95, -0.65, 0.55),
                lookat=(0.3, 0.05, 0.1),
                fov=45,
                GUI=False,
                debug=True,
            )

        # == wrist camera sensor (lazy rendering — zero cost until read()) ==
        if _ENABLE_MADRONA and gs.backend == gs.cuda:
            CameraOptions = BatchRendererCameraOptions
            cam_kwargs = dict(use_rasterizer=True)
        else:
            CameraOptions = RasterizerCameraOptions
            cam_kwargs = {}

        self.wrist_cam = self.scene.add_sensor(
            CameraOptions(
                res=self.wrist_cam_res,
                pos=(0.05864, 0.009, 0.05427),
                lookat=(0.156, 0.0, 0.0),
                fov=60,
                entity_idx=self.robot._robot_entity.idx,
                link_idx_local=self.robot._ee_link.idx_local,
                lights=[{"pos": (0.0, 0.0, 1.0), "color": (1.0, 1.0, 1.0), "intensity": 1.0}],
                **cam_kwargs,
            )
        )

        # == camera data readers ==
        def _read_scene_cam(cam):
            rgb = cam.render(rgb=True)[0]
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb[..., :3]

        def _read_sensor_cam(cam):
            return cam.read(envs_idx=0).rgb

        # == set up video recording (must be before build) ==
        record_video = env_cfg.get("record_video", {})
        for cam_name, filename in record_video.items():
            cam = getattr(self, cam_name)
            reader = _read_scene_cam if isinstance(cam, Camera) else _read_sensor_cam
            self.scene.add_recorder(
                data_func=partial(reader, cam),
                rec_options=gs.recorders.VideoFile(filename=filename, fps=15),
            )

        # build
        self.scene.build(n_envs=env_cfg["num_envs"], env_spacing=(2.0, 2.0))
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device, dtype=gs.tc_float)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.current_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.extras = dict()

    def get_active_object_state(self):
        return self.object.get_pos(), self.object.get_quat()

    def _reset_idx(self, envs_idx=None) -> None:
        """Reset specified environments.

        Parameters
        ----------
        envs_idx : torch.Tensor or None
            Boolean mask of shape (num_envs,) for selective reset, or None for full reset.
        """
        # Reset robot
        self.robot.reset(envs_idx)

        # Camera pose jitter
        dr_cfg = self.env_cfg.get("domain_randomization", {})
        if dr_cfg.get("randomize_camera_jitter", False):
            jitter_cfg = dr_cfg.get("camera_jitter", {})
            pos_std = jitter_cfg.get("pos_std", 0.0)
            quat_std = jitter_cfg.get("quat_std", 0.0)

            n_reset = self.num_envs if envs_idx is None else int(envs_idx.sum().item())

            if not hasattr(self, "cam_pos_offset"):
                self.cam_pos_offset = torch.zeros((self.num_envs, 3), device=self.device)
                self.cam_quat_offset = torch.zeros((self.num_envs, 4), device=self.device)
                self.cam_quat_offset[:, 0] = 1.0

            all_envs_list = list(range(self.num_envs))

            if pos_std > 0 and n_reset > 0:
                new_pos_offset = torch.randn((n_reset, 3), device=self.device) * pos_std
                if envs_idx is None:
                    self.cam_pos_offset.copy_(new_pos_offset)
                else:
                    self.cam_pos_offset[envs_idx] = new_pos_offset

                self.wrist_cam.set_pos_offset(self.cam_pos_offset, envs_idx=all_envs_list)

            if quat_std > 0 and n_reset > 0:
                xyz_noise = torch.randn((n_reset, 3), device=self.device) * quat_std
                w = torch.ones((n_reset, 1), device=self.device)
                new_quat_offset = torch.cat([w, xyz_noise], dim=-1)
                new_quat_offset = new_quat_offset / torch.norm(new_quat_offset, dim=-1, keepdim=True)

                if envs_idx is None:
                    self.cam_quat_offset.copy_(new_quat_offset)
                else:
                    self.cam_quat_offset[envs_idx] = new_quat_offset

                self.wrist_cam.set_quat_offset(self.cam_quat_offset, envs_idx=all_envs_list)

        # Generate random object state for all envs
        if "eval_cube_pos" in self.env_cfg:
            random_pos = torch.tensor(self.env_cfg["eval_cube_pos"], device=self.device).expand(self.num_envs, -1)
        else:
            (x_min, x_max), (y_min, y_max) = self.env_cfg["cube_x_range"], self.env_cfg["cube_y_range"]
            random_x = torch.rand(self.num_envs, device=self.device) * (x_max - x_min) + x_min
            random_y = torch.rand(self.num_envs, device=self.device) * (y_max - y_min) + y_min
            random_z = torch.full((self.num_envs,), self.cube_z_center, device=self.device)
            random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        random_yaw = torch.zeros(self.num_envs, device=self.device)
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        goal_pose = torch.cat([random_pos, goal_yaw], dim=-1)

        # Reset object — set_pos/set_quat with skip_forward, then FK runs once for everything
        if envs_idx is None:
            self.prev_actions.zero_()
            self.goal_pose.copy_(goal_pose)
            self.object.set_pos(random_pos, skip_forward=True)
            self.object.set_quat(goal_yaw, skip_forward=True)
            if hasattr(self, "finger_tip_vis"):
                self.finger_tip_vis.set_pos(self.robot.finger_tip_pose[:, :3], skip_forward=True)
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            self.prev_actions[envs_idx] = 0.0
            torch.where(envs_idx[:, None], goal_pose, self.goal_pose, out=self.goal_pose)
            self.object.set_pos(random_pos[envs_idx], envs_idx=envs_idx, skip_forward=True)
            self.object.set_quat(goal_yaw[envs_idx], envs_idx=envs_idx, skip_forward=True)
            if hasattr(self, "finger_tip_vis"):
                self.finger_tip_vis.set_pos(self.robot.finger_tip_pose[:, :3], envs_idx=envs_idx, skip_forward=True)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Invalidate camera cache after state change
        self.wrist_cam._stale = True

        # Fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

    def reset(self) -> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = torch.clamp(actions, -1.0, 1.0)

        # apply action
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions)
        self.scene.step()

        if hasattr(self, "finger_tip_vis"):
            self.finger_tip_vis.set_pos(self.robot.finger_tip_pose[:, :3], skip_forward=True)

        # update time
        self.episode_length_buf += 1

        # check termination (bool mask)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # timeout for value bootstrapping (only true timeouts, not NaN errors)
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self.current_actions = actions.clone()

        # compute reward before reset (reflects terminal state)
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        self.prev_actions = self.current_actions.clone()

        # soft-reset envs that need it
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        ee_pos, ee_quat = (
            self.robot.finger_tip_pose[:, :3],
            self.robot.finger_tip_pose[:, 3:7],
        )
        cube_pos, _ = self.get_active_object_state()

        obs_components = [
            ee_pos,  # finger tip position (3)
            ee_quat,  # finger tip orientation (w, x, y, z) (4)
            self._gripper_pos().unsqueeze(-1),  # gripper opening (1)
            cube_pos,  # cube position (3)
        ]
        self.obs_buf = torch.cat(obs_components, dim=-1)
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def get_student_state_obs(self) -> torch.Tensor:
        """State input of the image-based student policy: finger tip pose (7) + gripper opening (1)."""
        return torch.cat([self.robot.finger_tip_pose, self._gripper_pos().unsqueeze(-1)], dim=-1)

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def get_wrist_rgb_image(self, normalize: bool = True) -> torch.Tensor:
        rgb = self.wrist_cam.read().rgb  # (B, H, W, 3)
        rgb = rgb.permute(0, 3, 1, 2).float()  # (B, 3, H, W)
        if normalize:
            rgb = rgb / 255.0
        return rgb

    def _gripper_pos(self) -> torch.Tensor:
        q_pos = self.robot._robot_entity.get_qpos()
        return (q_pos[:, self.robot._left_finger_dof] + q_pos[:, self.robot._right_finger_dof]) / 2.0

    # ------------ begin reward functions----------------
    def _reward_reach_cube(self) -> torch.Tensor:
        # Reach the cube: 1 - tanh(d / sigma)
        finger_pos = self.robot.finger_tip_pose[:, :3]
        cube_pos = self.get_active_object_state()[0]
        pos_dist = torch.norm(finger_pos - cube_pos, p=2, dim=-1)
        reach_reward = 1.0 - torch.tanh(pos_dist / 0.3)

        # Gripper tracking: fully open at >= 10 cm from the cube, closing linearly to 15 mm per finger
        # at the cube (narrower than half the 5 cm cube so that the fingers squeeze it)
        target_finger_pos = 0.015 + (self.robot._gripper_open_dof - 0.015) * torch.clamp(pos_dist / 0.1, min=0.0, max=1.0)
        max_error = self.robot._gripper_open_dof - 0.015
        gripper_error = torch.abs(self._gripper_pos() - target_finger_pos) / max_error
        gripper_tracking_reward = 1.0 - torch.clamp(gripper_error, min=0.0, max=1.0)

        # Small weight on the gripper term: otherwise its penalty for a wrongly set gripper outweighs
        # the reach gradient, and the policy stops about 10 cm short of the cube.
        return reach_reward + 0.2 * gripper_tracking_reward

    def _reward_lift_cube(self) -> torch.Tensor:
        # Lift the cube to lift_height and hold it there
        z = self.get_active_object_state()[0][:, 2]

        # +1 bonus for lifting 5 mm, to help exploration
        bonus_reward = (z > self.cube_z_center + 0.005).to(dtype=gs.tc_float)

        height_reward = 1.0 - torch.tanh(torch.abs(z - self.env_cfg["lift_height"]) / 0.1)
        return height_reward + bonus_reward

    def _reward_action_rate(self) -> torch.Tensor:
        # Penalty: (act - prev_act)^2
        return -torch.sum(torch.square(self.current_actions - self.prev_actions), dim=-1)

    def _reward_joint_vel(self) -> torch.Tensor:
        # Penalty: joint_vel^2
        joint_vel = self.robot._robot_entity.get_dofs_velocity()
        return -torch.sum(torch.square(joint_vel), dim=-1)

    # ------------ end reward functions----------------


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid(gravity_compensation=1.0)
        morph: gs.morphs.MJCF = gs.morphs.MJCF(
            file=WXAI_MJCF,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._gripper_open_dof = 0.044
        self._gripper_close_dof = 0.00

        self._ik_method: Literal["gs_ik", "dls_ik"] = args["ik_method"]

        # == some buffer initialization ==
        self._init()

    def set_pd_gains(self):
        # 6 arm joints + 2 gripper fingers
        self._robot_entity.set_dofs_kp(
            torch.tensor([200.0, 200.0, 200.0, 100.0, 50.0, 50.0, 1000.0, 1000.0], device=self._device),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 50.0, 50.0], device=self._device),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-27.0, -27.0, -27.0, -7.0, -7.0, -7.0, -400.0, -400.0], device=self._device),
            torch.tensor([27.0, 27.0, 27.0, 7.0, 7.0, 7.0, 400.0, 400.0], device=self._device),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._args["default_arm_dof"] + self._args["default_gripper_dof"]
        self._init_qpos = torch.tensor(self._default_joint_angles, dtype=torch.float32, device=self._device)
        # On MPS/Metal, batched linear algebra is extremely slow due to per-element kernel dispatch.
        # Running the DLS solve on CPU is ~300x faster in that case.
        self._dls_solve_on_cpu = self._device == "mps" or str(self._device).startswith("mps")
        dls_lam_device = "cpu" if self._dls_solve_on_cpu else self._device
        self._dls_lambda_matrix = (0.01**2) * torch.eye(6, device=dls_lam_device)

        # Joint position commands; actions are integrated onto these, not onto the measured joint positions
        self.cmd_qpos = self._init_qpos.clone().expand(self._num_envs, -1).contiguous()

    def reset(self, envs_idx=None, skip_forward=True):
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )
        if envs_idx is None:
            self.cmd_qpos[:] = self._init_qpos
        else:
            self.cmd_qpos[envs_idx] = self._init_qpos
        self._robot_entity.control_dofs_position(
            self._init_qpos,
            envs_idx=envs_idx,
        )

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply the action to the robot."""
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")

        # the 7th action dimension is a delta of the gripper opening
        gripper_delta = action[:, 6]
        current_gripper = self.cmd_qpos[:, self._left_finger_dof]
        gripper_target = torch.clamp(current_gripper + gripper_delta, self._gripper_close_dof, self._gripper_open_dof)
        q_pos[:, self._left_finger_dof] = gripper_target
        q_pos[:, self._right_finger_dof] = gripper_target

        self.cmd_qpos = q_pos.clone()
        self._robot_entity.control_dofs_position(q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped least squares inverse kinematics.

        Solves (J @ J^T + lambda^2 * I) @ y = dx, then dq = J^T @ y.
        """
        delta_pose = action[:, :6]
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        if self._dls_solve_on_cpu:
            jacobian = jacobian.cpu()
            delta_pose = delta_pose.cpu()
        A = torch.baddbmm(self._dls_lambda_matrix, jacobian, jacobian.mT)
        y = torch.linalg.solve(A, delta_pose)
        delta_joint_pos = (jacobian.mT @ y.unsqueeze(-1)).squeeze(-1)
        if self._dls_solve_on_cpu:
            delta_joint_pos = delta_joint_pos.to(self._device)
        return self.cmd_qpos + delta_joint_pos

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = self._right_finger_link.get_pos(), self._right_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        The center finger pose is the average of the left and right finger poses.
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)

    @property
    def finger_tip_pose(self) -> torch.Tensor:
        """
        The finger tip pose: the center finger pose shifted ~7 cm along its +X axis (towards the finger tips).
        """
        center_pose = self.center_finger_pose
        local_offset = torch.tensor([0.0695, 0.0, 0.0], device=self._device).unsqueeze(0).repeat(self._num_envs, 1)
        tip_pos = transform_by_trans_quat(local_offset, center_pose[:, :3], center_pose[:, 3:7])
        return torch.cat([tip_pos, center_pose[:, 3:7]], dim=-1)
