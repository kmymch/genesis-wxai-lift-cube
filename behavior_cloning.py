import os
import time
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as v2


class BehaviorCloning:
    """Behavior cloning (DAgger) of the RL teacher into a wrist-camera student, with cube position estimation."""

    def __init__(self, env, cfg: dict, teacher: nn.Module, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher
        self._num_steps_per_env = cfg["num_steps_per_env"]

        wrist_shape = (3, env.wrist_cam_res[1], env.wrist_cam_res[0])
        action_dim = env.num_actions

        # Policy with an action head and a cube position head
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])

        # Exponential moving average of the weights, used for evaluation and saved checkpoints
        self._ema = AveragedModel(self._policy, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True)

        # Experience buffer with cube position data
        self._buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=self._cfg["buffer_size"],
            wrist_shape=wrist_shape,
            state_dim=self._cfg["policy"]["action_head"]["state_obs_dim"],
            action_dim=action_dim,
            device=device,
            dtype=self._policy.dtype,
        )

        # Training state
        self._current_iter = 0

        # Color Jitter setup for batch augmentation
        self.color_jitter = None
        dr_cfg = env.env_cfg.get("domain_randomization", {})
        if dr_cfg.get("randomize_color_jitter", False):
            jitter_cfg = dr_cfg.get("color_jitter_params", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05})
            self.color_jitter = v2.ColorJitter(**jitter_cfg)

    def learn(self, num_learning_iterations: int, log_dir: str) -> None:
        self._buffer.clear()

        tf_writer = SummaryWriter(log_dir)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=num_learning_iterations, eta_min=self._cfg["min_learning_rate"]
        )

        for it in range(num_learning_iterations):
            # Collect experience
            start_time = time.time()
            self._collect_with_rl_teacher()
            end_time = time.time()
            forward_time = end_time - start_time

            # Training steps for both action and cube position prediction
            total_action_loss = 0.0
            total_pose_loss = 0.0
            total_pos_err = 0.0
            num_batches = 0

            start_time = time.time()
            generator = self._buffer.get_batches(self._cfg.get("num_mini_batches", 4), self._cfg["num_epochs"])
            for batch in generator:
                # Apply ColorJitter to mini-batch on the fly
                wrist_obs_batch = batch["wrist_obs"]
                if self.color_jitter is not None:
                    wrist_obs_batch = self.color_jitter(wrist_obs_batch)

                # Forward pass for both action and cube position prediction
                pred_action = self._policy(wrist_obs_batch, batch["robot_pose"])
                pred_pos = self._policy.predict_cube_pos(wrist_obs_batch)

                # Compute action prediction loss
                action_loss = F.mse_loss(pred_action, batch["actions"])

                # Cube position loss (position relative to the finger tip)
                pose_loss = self._cfg["pose_loss_weight"] * F.mse_loss(pred_pos, batch["cube_rel_pos"])

                # Combined loss with weights
                total_loss = action_loss + pose_loss

                # Backward pass
                self._optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])
                self._optimizer.step()
                self._ema.update_parameters(self._policy)

                total_action_loss += action_loss
                total_pose_loss += pose_loss
                with torch.no_grad():
                    total_pos_err += (pred_pos - batch["cube_rel_pos"]).norm(dim=-1).mean()
                num_batches += 1

            end_time = time.time()
            backward_time = end_time - start_time
            scheduler.step()

            # Compute average losses
            if num_batches == 0:
                raise ValueError("No batches collected")
            else:
                avg_action_loss = total_action_loss / num_batches
                avg_pose_loss = total_pose_loss / num_batches
                avg_pos_err_cm = 100 * total_pos_err / num_batches

            self._current_iter = it
            fps = (self._num_steps_per_env * self._env.num_envs) / forward_time
            # Logging
            if (it + 1) % self._cfg["log_freq"] == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]

                tf_writer.add_scalar("loss/action_loss", avg_action_loss, it)
                tf_writer.add_scalar("loss/pose_loss", avg_pose_loss, it)
                tf_writer.add_scalar("loss/total_loss", avg_action_loss + avg_pose_loss, it)
                tf_writer.add_scalar("cube_pos_error_cm", avg_pos_err_cm, it)
                tf_writer.add_scalar("lr", current_lr, it)
                tf_writer.add_scalar("buffer_size", self._buffer.size, it)
                tf_writer.add_scalar("speed/forward", forward_time, it)
                tf_writer.add_scalar("speed/backward", backward_time, it)
                tf_writer.add_scalar("speed/fps", int(fps), it)

                print("--------------------------------")
                info_str = f" | Iteration:     {it + 1:04d}\n"
                info_str += f" | Action Loss:   {avg_action_loss:.6f}\n"
                info_str += f" | Pose Loss:     {avg_pose_loss:.6f}\n"
                info_str += f" | Cube Pos Err:  {avg_pos_err_cm:.1f} cm\n"
                info_str += f" | Total Loss:    {avg_action_loss + avg_pose_loss:.6f}\n"
                info_str += f" | Learning Rate: {current_lr:.6f}\n"
                info_str += f" | Forward Time:  {forward_time:.2f}s\n"
                info_str += f" | Backward Time: {backward_time:.2f}s\n"
                info_str += f" | FPS:           {int(fps)}"
                print(info_str)

            # Save checkpoints periodically
            if (it + 1) % self._cfg["save_freq"] == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it + 1:04d}.pt"))

            # Evaluate student policy periodically
            if (it + 1) % self._cfg.get("eval_freq", 50) == 0:
                self._evaluate_student(tf_writer, it)

        tf_writer.close()

    def _student_prob(self) -> float:
        """Probability of executing the student's action, ramped up linearly over the first iterations."""
        ramp = self._cfg["dagger_ramp_iters"]
        return self._cfg["dagger_max_student_prob"] * min(1.0, self._current_iter / ramp)

    def _collect_with_rl_teacher(self) -> None:
        """Collect experience from environment using wrist rgb images and cube positions."""
        # Get state observation
        obs_dict = self._env.get_observations()
        with torch.inference_mode():
            for _ in range(self._num_steps_per_env):
                wrist_obs = self._env.get_wrist_rgb_image(normalize=True)

                # Get teacher action, clipped to the range the environment applies
                teacher_action = torch.clamp(self._teacher(obs_dict).detach(), -1.0, 1.0)

                state_obs = self._env.get_student_state_obs()
                cube_rel_pos = self._env.get_active_object_state()[0] - self._env.robot.finger_tip_pose[:, :3]

                # Store in buffer (saving clean images)
                self._buffer.add(wrist_obs, state_obs, cube_rel_pos, teacher_action)

                # Apply jitter for student action during rollout
                jittered_wrist_obs = wrist_obs.clone()
                if self.color_jitter is not None:
                    jittered_wrist_obs = self.color_jitter(jittered_wrist_obs)

                # Step environment with student action (eval mode: BatchNorm running stats)
                self._policy.eval()
                student_action = self._policy(jittered_wrist_obs.float(), state_obs.float())
                self._policy.train()

                # DAgger: execute the student's action with probability student_prob, otherwise the teacher's.
                # Labels are always the teacher's actions, so the student learns to recover from its own mistakes.
                use_student = torch.rand(self._env.num_envs, 1, device=self._device) < self._student_prob()
                action = torch.where(use_student, student_action, teacher_action)

                obs_dict, _, _, _ = self._env.step(action)

    @torch.inference_mode()
    def _evaluate_student(self, tf_writer: SummaryWriter, iteration: int) -> None:
        """Evaluate the pure student policy for one full episode."""
        self._env.reset()
        total_reward = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)

        print(" | Running pure student evaluation rollout...")
        ema_policy = self._ema.module.eval()
        for _ in range(self._env.max_episode_length):
            wrist_obs = self._env.get_wrist_rgb_image(normalize=True)
            student_action = ema_policy(wrist_obs.float(), self._env.get_student_state_obs().float())
            _, reward, _, _ = self._env.step(student_action)
            total_reward += reward

        mean_reward = total_reward.mean().item()
        tf_writer.add_scalar("eval/reward_mean", mean_reward, iteration)
        print(f" | [Eval] Iteration {iteration + 1:04d} - Mean Reward: {mean_reward:.4f}")

        # Manually extract and log reward components from episode_sums to ensure
        # we get the exact accumulated rewards up to max_episode_length
        for key, value in self._env.episode_sums.items():
            mean_val = value.mean().item() / self._env.env_cfg["episode_length_s"]
            tf_writer.add_scalar(f"eval/rew_{key}", mean_val, iteration)

        # Reset environment again to clear evaluation state and resume training cleanly
        self._env.reset()

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self._ema.module.state_dict(),  # EMA weights, used for inference
            "online_model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self._current_iter,
            "config": self._cfg,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint (EMA weights)."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["current_iter"]
        print(f"Model loaded from {path}")


class ExperienceBuffer:
    """A first-in-first-out buffer for experience replay."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        wrist_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ):
        self._num_envs = num_envs
        self._max_size = max_size
        self._wrist_shape = wrist_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._ptr = 0
        self._size = 0

        # Buffers for data
        self._wrist_obs = torch.empty(max_size, num_envs, *wrist_shape, dtype=dtype, device=device)
        self._robot_pose = torch.empty(max_size, num_envs, state_dim, dtype=dtype, device=device)
        self._cube_rel_pos = torch.empty(max_size, num_envs, 3, dtype=dtype, device=device)
        self._actions = torch.empty(max_size, num_envs, action_dim, dtype=dtype, device=device)

    def add(
        self,
        wrist_obs: torch.Tensor,
        robot_pose: torch.Tensor,
        cube_rel_pos: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """Add experience to buffer."""
        self._wrist_obs[self._ptr] = wrist_obs
        self._robot_pose[self._ptr] = robot_pose
        self._cube_rel_pos[self._ptr] = cube_rel_pos
        self._actions[self._ptr] = actions
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def get_batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[dict[str, torch.Tensor]]:
        """Generate batches for training."""
        # calculate the size of each mini-batch
        batch_size = self._size // num_mini_batches
        for _ in range(num_epochs):
            indices = torch.randperm(self._size, device=self._device)
            for batch_idx in range(0, self._size, batch_size):
                batch_indices = indices[batch_idx : batch_idx + batch_size]

                # Yield a mini-batch of data
                yield {
                    "wrist_obs": self._wrist_obs[batch_indices].reshape(-1, *self._wrist_shape),
                    "robot_pose": self._robot_pose[batch_indices].reshape(-1, self._state_dim),
                    "cube_rel_pos": self._cube_rel_pos[batch_indices].reshape(-1, 3),
                    "actions": self._actions[batch_indices].reshape(-1, self._action_dim),
                }

    def clear(self) -> None:
        """Clear the buffer."""
        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self._size


class Policy(nn.Module):
    """Wrist-camera student policy.

    A CNN estimates the cube position relative to the finger tip from the wrist image, and an MLP maps
    the robot state and that estimate to the action, i.e. the same inputs as the teacher.
    """

    def __init__(self, config: dict, action_dim: int):
        super().__init__()

        self.wrist_encoder = self._build_cnn(config["vision_encoder"])
        vision_encoder_output_dim = config["vision_encoder"]["conv_layers"][-1]["out_channels"] * 4 * 4

        # MLP for cube position prediction
        pose_mlp_cfg = config["pose_head"]
        pose_mlp_cfg["input_dim"] = vision_encoder_output_dim
        pose_mlp_cfg["output_dim"] = 3
        self.pose_mlp = self._build_mlp(pose_mlp_cfg)

        # MLP for action prediction from the state and the estimated cube position
        mlp_cfg = config["action_head"]
        self.state_obs_dim = mlp_cfg["state_obs_dim"]
        mlp_cfg["input_dim"] = self.state_obs_dim + 3
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

    @property
    def dtype(self):
        """Get the dtype of the policy's parameters."""
        return next(self.parameters()).dtype

    @staticmethod
    def _build_cnn(config: dict) -> nn.Sequential:
        """Build CNN encoder for rgb images."""
        layers = []

        # Build layers from configuration
        for conv_config in config["conv_layers"]:
            layers.extend(
                [
                    nn.Conv2d(
                        conv_config["in_channels"],
                        conv_config["out_channels"],
                        kernel_size=conv_config["kernel_size"],
                        stride=conv_config["stride"],
                        padding=conv_config["padding"],
                    ),
                    nn.BatchNorm2d(conv_config["out_channels"]),
                    nn.ReLU(),
                ]
            )

        # Add adaptive pooling if specified
        if config.get("pooling") == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp(config: dict) -> nn.Sequential:
        mlp_input_dim = config["input_dim"]
        layers = []
        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def predict_cube_pos(self, wrist_obs: torch.Tensor) -> torch.Tensor:
        """Predict the cube position relative to the finger tip from the wrist rgb image."""
        return self.pose_mlp(self.wrist_encoder(wrist_obs).flatten(start_dim=1))

    def forward(self, wrist_obs: torch.Tensor, state_obs: torch.Tensor) -> torch.Tensor:
        # detached so that only the position loss trains the encoder and the position head
        cube_rel_pos = self.predict_cube_pos(wrist_obs).detach()
        return self.mlp(torch.cat([state_obs, cube_rel_pos], dim=-1))
