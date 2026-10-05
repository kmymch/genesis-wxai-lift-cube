import argparse
import math
import re
import pickle
from importlib import metadata
from pathlib import Path

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner
from behavior_cloning import BehaviorCloning

import genesis as gs

from env import GraspEnv


def get_train_cfg(exp_name):
    # stage 1: privileged reinforcement learning
    rl_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    # stage 2: vision-based behavior cloning
    bc_cfg_dict = {
        # Basic training parameters
        "num_steps_per_env": 24,
        "learning_rate": 1e-4,  # decayed to min_learning_rate with a cosine schedule
        "min_learning_rate": 1e-5,
        "ema_decay": 0.995,  # per gradient step; about 20 steps per iteration
        "num_epochs": 5,
        "num_mini_batches": 4,
        "max_grad_norm": 1.0,
        # weight of the cube position loss (MSE in m^2); 100 makes a 1 cm error comparable to the action loss
        "pose_loss_weight": 100.0,
        # DAgger: probability of executing the student's action, ramped from 0 over dagger_ramp_iters
        "dagger_max_student_prob": 0.5,
        "dagger_ramp_iters": 100,
        # Network architecture
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,  # 3 channel for rgb image
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
                "pooling": "adaptive_avg",
            },
            "action_head": {
                "state_obs_dim": 8,  # finger tip pose (7) + gripper opening (1)
                "hidden_dims": [256, 256, 128],
            },
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        # Training settings
        "buffer_size": 1000,
        "log_freq": 1,
        "save_freq": 50,
        "eval_freq": 10,
    }

    return rl_cfg_dict, bc_cfg_dict


def get_task_cfgs():
    env_cfg = {
        "num_envs": 10,
        "num_actions": 7,
        # delta per step: position (m), orientation (rad), gripper opening (m)
        "action_scales": [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.005],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.05, 0.05, 0.05],
        "cube_x_range": [0.3, 0.55],
        "cube_y_range": [0.0, 0.2],
        "lift_height": 0.2,  # m, height of the cube center to hold
        "wrist_cam_resolution": (128, 96),
        "visualize_camera": False,
        "show_visual_helpers": True,
        "domain_randomization": {
            "randomize_camera_jitter": True,
            "camera_jitter": {
                "pos_std": 0.005,  # 5 mm
                "quat_std": 0.02,  # std of the quaternion vector part; about 0.04 rad per axis
            },
            # applied to the wrist camera images in behavior cloning
            "randomize_color_jitter": True,
            "color_jitter_params": {
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.3,
                "hue": 0.1,
            },
        },
    }
    reward_scales = {
        "reach_cube": 1.0,
        "lift_cube": 2.0,
        "action_rate": 0.05,
        "joint_vel": 0.05,
    }
    # wxai follower specific
    robot_cfg = {
        "ee_link_name": "link_6",
        "gripper_link_names": ["carriage_left", "carriage_right"],
        "default_arm_dof": [0.0, math.pi / 6, math.pi / 6, 0.0, 0.0, 0.0],
        "default_gripper_dof": [0.044, 0.044],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_scales, robot_cfg


def load_teacher_policy(env, rl_train_cfg, exp_name):
    # load teacher policy
    log_dir = Path("logs") / f"{exp_name + '_' + 'rl'}"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    teacher_policy = runner.get_inference_policy(device=gs.device)
    return teacher_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="lift_cube")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=None, help="default: 1000 for rl, 400 for bc")
    parser.add_argument("--stage", type=str, default="rl", choices=["rl", "bc"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.max_iterations is None:
        args.max_iterations = 1000 if args.stage == "rl" else 400

    # === init ===
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    # === task cfgs and trainning algos cfgs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    # the finger tip marker must not appear in the student's camera images
    env_cfg["show_visual_helpers"] = args.stage != "bc"
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name)

    # === log dir ===
    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # === env ===
    # BC only needs a small number of envs, e.g., 10
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10
    env_cfg["box_fixed"] = False

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === runner ===
    if args.stage == "bc":
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=log_dir)
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training

# to train the RL policy
python train.py --stage=rl

# to train the BC policy (requires RL policy to be trained first)
python train.py --stage=bc
"""
