import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import panda_gym  # required so PandaPush-v3 is registered

from stable_baselines3 import PPO, SAC


def load_model(model_path: str, env=None):
    lower_path = model_path.lower()

    if "ppo" in lower_path:
        return PPO.load(model_path)

    if "sac" in lower_path:
        return SAC.load(model_path, env=env)

    raise ValueError("Could not infer PPO or SAC from the model filename.")


def reset_env(env, seed: int):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env.reset(seed=seed)


def short_model_name(model_path: str) -> str:
    stem = Path(model_path).stem
    parts = stem.split("_")

    if len(parts) >= 5 and parts[1] == "push":
        algo = parts[0]
        strategy = parts[2]
        train_env = parts[3]
        timesteps = parts[4]

        if strategy == "none":
            return f"{algo}_{train_env}_{timesteps}"

        return f"{algo}_{strategy}_{train_env}_{timesteps}"

    return stem


def render_start_finish(model_path: str, env_type: str, seed: int, out_dir: str) -> Path:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=env_type,
        reward_type="dense",
    )
    model = load_model(model_path, env=env)

    obs, _ = reset_env(env, seed)
    start_frame = env.render()

    terminated = False
    truncated = False
    finish_frame = start_frame

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        finish_frame = env.render()

    env.close()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"part2_{short_model_name(model_path)}_{env_type}_seed{seed}_start_finish.png"
    save_path = out_path / filename

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, frame, title in zip(axes, [start_frame, finish_frame], ["Start", "Finish"]):
        ax.imshow(np.asarray(frame))
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PandaPush start and finish frames")
    parser.add_argument("--model-path", required=True, help="Path to a PPO/SAC model zip file")
    parser.add_argument(
        "--env-type",
        default="target",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument("--seed", type=int, default=53, help="Evaluation seed")
    parser.add_argument("--out-dir", default="../renders", help="Directory for the output PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = render_start_finish(
        model_path=args.model_path,
        env_type=args.env_type,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(f"Saved render image to: {save_path}")


if __name__ == "__main__":
    main()
