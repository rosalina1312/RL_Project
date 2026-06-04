import argparse
import json
import os
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import numpy as np
import panda_gym  # required so PandaPush-v3 is registered
import stable_baselines3 as sb3
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rand_wrapper import RandomizationWrapper


def float_to_tag(value: float) -> str:
    return str(value).replace(".", "p")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def reset_env(env, seed: int):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env.reset(seed=seed)


def get_runtime_versions() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "gymnasium_version": gym.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": sb3.__version__,
        "panda_gym_version": getattr(panda_gym, "__version__", "local"),
    }


def save_training_metadata(args, run_name: str, save_name: str) -> str:
    metadata_path = f"{save_name}_metadata.json"
    metadata = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "run_name": run_name,
        "model_path": f"{save_name}.zip",
        "args": vars(args),
        "runtime_versions": get_runtime_versions(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or SAC on PandaPush-v3")

    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["ppo", "sac"],
        help="RL algorithm to train",
    )

    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )

    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Number of training timesteps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor",
    )

    parser.add_argument(
        "--ppo-n-steps",
        type=int,
        default=1024,
        help="Number of PPO rollout steps",
    )

    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=64,
        help="PPO batch size",
    )

    parser.add_argument(
        "--sac-buffer-size",
        type=int,
        default=100_000,
        help="SAC replay buffer size",
    )

    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=256,
        help="SAC batch size",
    )

    parser.add_argument(
        "--mass-min",
        type=float,
        default=0.5,
        help="Minimum object mass for UDR and ADR lower limit",
    )

    parser.add_argument(
        "--mass-max",
        type=float,
        default=4.0,
        help="Maximum object mass for UDR",
    )

    parser.add_argument(
        "--adr-initial-min",
        type=float,
        default=0.8,
        help="Initial minimum mass for ADR",
    )

    parser.add_argument(
        "--adr-initial-max",
        type=float,
        default=1.2,
        help="Initial maximum mass for ADR",
    )

    parser.add_argument(
        "--adr-max-limit",
        type=float,
        default=4.0,
        help="Maximum allowed mass for ADR expansion",
    )

    parser.add_argument(
        "--adr-step",
        type=float,
        default=0.2,
        help="ADR range expansion step",
    )

    parser.add_argument(
        "--adr-success-threshold",
        type=float,
        default=0.7,
        help="Recent success-rate threshold required to expand ADR range",
    )

    parser.add_argument(
        "--adr-window-size",
        type=int,
        default=20,
        help="Number of recent episodes used to decide ADR expansion",
    )

    parser.add_argument(
        "--print-masses",
        action="store_true",
        help="Print sampled cube masses during randomized training",
    )

    parser.add_argument(
        "--load-model-path",
        type=str,
        default=None,
        help="Optional PPO/SAC checkpoint to continue training from",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional exact run name used for logs and model saving",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce Stable-Baselines3 training logs",
    )

    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the Stable-Baselines3 progress bar",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="faiml-rl-panda-push",
        help="WandB project name",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    if args.sampling_strategy != "none":
        env = RandomizationWrapper(
            env,
            mode=args.sampling_strategy,
            mass_range=(args.mass_min, args.mass_max),
            adr_initial_range=(args.adr_initial_min, args.adr_initial_max),
            adr_min_limit=args.mass_min,
            adr_max_limit=args.adr_max_limit,
            adr_step=args.adr_step,
            adr_success_threshold=args.adr_success_threshold,
            adr_window_size=args.adr_window_size,
            verbose=args.print_masses,
            seed=args.seed,
        )

    env = Monitor(env)
    reset_env(env, args.seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)
    os.makedirs("wandb_models", exist_ok=True)

    lr_tag = float_to_tag(args.learning_rate)
    gamma_tag = float_to_tag(args.gamma)

    if args.algo == "ppo":
        hp_tag = (
            f"lr{lr_tag}_g{gamma_tag}_"
            f"n{args.ppo_n_steps}_b{args.ppo_batch_size}"
        )
    else:
        hp_tag = (
            f"lr{lr_tag}_g{gamma_tag}_"
            f"buf{args.sac_buffer_size}_b{args.sac_batch_size}"
        )

    if args.sampling_strategy == "udr":
        rand_tag = (
            f"m{float_to_tag(args.mass_min)}-"
            f"{float_to_tag(args.mass_max)}"
        )
    elif args.sampling_strategy == "adr":
        rand_tag = (
            f"init{float_to_tag(args.adr_initial_min)}-"
            f"{float_to_tag(args.adr_initial_max)}_"
            f"lim{float_to_tag(args.mass_min)}-"
            f"{float_to_tag(args.adr_max_limit)}_"
            f"step{float_to_tag(args.adr_step)}_"
            f"thr{float_to_tag(args.adr_success_threshold)}_"
            f"win{args.adr_window_size}"
        )
    else:
        rand_tag = "fixed"

    run_name = (
        f"{args.algo}_push_"
        f"{args.sampling_strategy}_{args.env_type}_"
        f"{args.timesteps // 1000}k_"
        f"{hp_tag}_{rand_tag}_seed{args.seed}"
    )
    if args.run_name is not None:
        run_name = args.run_name

    save_name = f"models/{run_name}"

    model_verbose = 0 if args.quiet else 1

    if args.load_model_path is not None:
        if args.algo == "ppo":
            model = PPO.load(args.load_model_path, env=env)
        else:
            model = SAC.load(args.load_model_path, env=env)
        model.verbose = model_verbose
        if args.quiet:
            model.set_logger(configure(folder=None, format_strings=[]))

    elif args.algo == "ppo":
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=model_verbose,
            learning_rate=args.learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            gamma=args.gamma,
            seed=args.seed,
            tensorboard_log="./tb_logs/",
        )

    elif args.algo == "sac":
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=model_verbose,
            learning_rate=args.learning_rate,
            buffer_size=args.sac_buffer_size,
            batch_size=args.sac_batch_size,
            gamma=args.gamma,
            seed=args.seed,
            tensorboard_log="./tb_logs/",
        )

    else:
        raise ValueError("Unknown algorithm. Use 'ppo' or 'sac'.")

    model.set_random_seed(args.seed)

    print("\nTraining configuration")
    print("Run name:", run_name)
    print("Algorithm:", args.algo)
    print("Environment type:", args.env_type)
    print("Sampling strategy:", args.sampling_strategy)
    print("Timesteps:", args.timesteps)
    print("Seed:", args.seed)
    print("Learning rate:", args.learning_rate)
    print("Gamma:", args.gamma)

    if args.algo == "ppo":
        print("PPO n_steps:", args.ppo_n_steps)
        print("PPO batch size:", args.ppo_batch_size)
    else:
        print("SAC buffer size:", args.sac_buffer_size)
        print("SAC batch size:", args.sac_batch_size)

    if args.sampling_strategy == "udr":
        print("UDR mass range:", (args.mass_min, args.mass_max))

    if args.sampling_strategy == "adr":
        print("ADR initial range:", (args.adr_initial_min, args.adr_initial_max))
        print("ADR limits:", (args.mass_min, args.adr_max_limit))
        print("ADR step:", args.adr_step)
        print("ADR success threshold:", args.adr_success_threshold)
        print("ADR window size:", args.adr_window_size)

    print("Save path:", save_name)
    print("Use WandB:", args.use_wandb)
    if args.load_model_path is not None:
        print("Continuing from:", args.load_model_path)

    callback = None
    wandb_run = None

    if args.use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "run_name": run_name,
                "algorithm": args.algo,
                "env_type": args.env_type,
                "sampling_strategy": args.sampling_strategy,
                "timesteps": args.timesteps,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "ppo_n_steps": args.ppo_n_steps,
                "ppo_batch_size": args.ppo_batch_size,
                "sac_buffer_size": args.sac_buffer_size,
                "sac_batch_size": args.sac_batch_size,
                "reward_type": "dense",
                "mass_min": args.mass_min,
                "mass_max": args.mass_max,
                "adr_initial_min": args.adr_initial_min,
                "adr_initial_max": args.adr_initial_max,
                "adr_max_limit": args.adr_max_limit,
                "adr_step": args.adr_step,
                "adr_success_threshold": args.adr_success_threshold,
                "adr_window_size": args.adr_window_size,
            },
            sync_tensorboard=True,
        )

        callback = WandbCallback(
            model_save_path=f"wandb_models/{run_name}",
            model_save_freq=20_000,
            verbose=2,
        )

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
        reset_num_timesteps=args.load_model_path is None,
        tb_log_name=run_name,
        callback=callback,
    )

    model.save(save_name)
    metadata_path = save_training_metadata(args, run_name, save_name)
    env.close()

    print("\nTraining finished")
    print(f"Saved model to: {save_name}.zip")
    print(f"Saved metadata to: {metadata_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
