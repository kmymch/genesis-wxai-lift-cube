import argparse
import copy
import re
import pickle
from importlib import metadata
from pathlib import Path

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import GraspEnv
from behavior_cloning import Policy


def load_rl_policy(env, train_cfg, log_dir):
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    """Load behavior cloning policy (EMA weights)."""
    policy = Policy(copy.deepcopy(bc_cfg["policy"]), env.num_actions).to(gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    print(f"Loaded BC checkpoint from {last_ckpt}")
    policy.load_state_dict(torch.load(last_ckpt, map_location=gs.device, weights_only=False)["model_state_dict"])

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="lift_cube")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize with UI")
    parser.add_argument(
        "-B", "--num_envs", type=int, default=1, help="Number of parallel episodes; reports the success rate"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the cube positions")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the scene and the policy cameras as video during evaluation",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Filename of the scene video under the log directory (default: video.mp4)",
    )
    parser.add_argument(
        "--cube_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Initial cube position (default: random, as in training)",
    )
    args = parser.parse_args()

    gs.init(seed=args.seed)
    torch.manual_seed(args.seed)

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)

    env_cfg["visualize_camera"] = args.vis or args.record
    env_cfg["num_envs"] = args.num_envs
    env_cfg["box_fixed"] = False
    env_cfg["show_visual_helpers"] = args.stage != "bc"
    # evaluate with the nominal wrist camera pose (no training-time jitter)
    env_cfg["domain_randomization"]["randomize_camera_jitter"] = False
    if args.cube_pos is not None:
        env_cfg["eval_cube_pos"] = args.cube_pos

    if args.record:
        env_cfg["record_video"] = {
            "vis_cam": str(log_dir / (args.video_path or "video.mp4")),
            "wrist_cam": str(log_dir / "wrist_cam.mp4"),
        }

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    obs_dict = env.reset()

    max_z = torch.zeros(env.num_envs, device=gs.device)
    with torch.no_grad():
        for _ in range(env.max_episode_length):
            if args.stage == "rl":
                actions = policy(obs_dict)
            else:
                actions = policy(env.get_wrist_rgb_image(normalize=True), env.get_student_state_obs())
            obs_dict, rews, dones, infos = env.step(actions)
            max_z = torch.maximum(max_z, env.get_active_object_state()[0][:, 2])

    if args.record:
        env.scene.stop_recording()

    final_z = env.get_active_object_state()[0][:, 2]
    if env.num_envs == 1:
        print(f"Cube height: max {max_z[0] * 100:.1f} cm, final {final_z[0] * 100:.1f} cm (goal {env_cfg['lift_height'] * 100:.0f} cm)")
    else:
        # success: the cube is held above 15 cm at the end of the episode
        success = (final_z > 0.15).float().mean().item()
        print(f"Success rate: {success * 100:.0f}% of {env.num_envs} episodes (cube above 15 cm at the end)")


if __name__ == "__main__":
    main()

"""
# evaluation
# For reinforcement learning model:
python eval.py --stage=rl

# For behavior cloning model:
python eval.py --stage=bc

# With video recording:
python eval.py --stage=bc --record

# Success rate over 300 episodes:
python eval.py --stage=bc -B 300
"""
