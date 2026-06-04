import argparse
import csv
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import panda_gym  # required so PandaPush-v3 is registered

from stable_baselines3 import PPO, SAC


def reset_env(env, seed: int):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env.reset(seed=seed)


def load_model(model_path: str, env=None):
    lower_path = model_path.lower()

    if "ppo" in lower_path:
        return PPO.load(model_path)

    if "sac" in lower_path:
        return SAC.load(model_path, env=env)

    raise ValueError(
        "Could not infer algorithm from model path. "
        "Please use a filename containing 'ppo' or 'sac'."
    )


def set_object_mass(env, mass: float) -> None:
    """
    Manually set the PandaPush cube mass.
    Used for robustness evaluation across different masses.
    """
    sim = env.unwrapped.task.sim
    object_body_id = sim._bodies_idx["object"]

    sim.physics_client.changeDynamics(
        bodyUniqueId=object_body_id,
        linkIndex=-1,
        mass=float(mass),
    )


def infer_algorithm(model_path: str) -> str:
    lower_path = model_path.lower()

    if "ppo" in lower_path:
        return "ppo"

    if "sac" in lower_path:
        return "sac"

    return "unknown"


def infer_sampling_strategy(model_path: str) -> str:
    lower_path = model_path.lower()

    if "_udr_" in lower_path:
        return "udr"

    if "_adr_" in lower_path:
        return "adr"

    if "_none_" in lower_path:
        return "none"

    return "unknown"


def append_results_to_csv(
    csv_path: str,
    experiment_name: str,
    model_path: str,
    env_type: str,
    eval_mass,
    n_episodes: int,
    seed: int,
    deterministic: bool,
    mean_return: float,
    std_return: float,
    min_return: float,
    max_return: float,
    success_rate,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "experiment_name",
        "model_name",
        "model_path",
        "algorithm",
        "sampling_strategy",
        "env_type",
        "eval_mass",
        "episodes",
        "seed",
        "deterministic",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "success_rate",
    ]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "algorithm": infer_algorithm(model_path),
        "sampling_strategy": infer_sampling_strategy(model_path),
        "env_type": env_type,
        "eval_mass": "" if eval_mass is None else eval_mass,
        "episodes": n_episodes,
        "seed": seed,
        "deterministic": deterministic,
        "mean_return": mean_return,
        "std_return": std_return,
        "min_return": min_return,
        "max_return": max_return,
        "success_rate": "" if success_rate is None else success_rate,
    }

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def evaluate(
    model_path: str,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    render_sleep: float,
    start_hold: float,
    end_hold: float,
    env_type: str,
    eval_mass,
    seed: int,
    save_csv: bool,
    csv_path: str,
    experiment_name: str,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"

    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        type=env_type,
        reward_type="dense",
    )

    model = load_model(model_path, env=env)

    episode_returns = []
    successes = []
    episode_lengths = []
    start_distances = []
    final_distances = []
    object_moves = []

    for episode in range(1, n_episodes + 1):
        obs, info = reset_env(env, seed + episode - 1)
        start_achieved_goal = np.array(obs["achieved_goal"], dtype=np.float32)
        desired_goal = np.array(obs["desired_goal"], dtype=np.float32)
        start_distance = float(np.linalg.norm(start_achieved_goal - desired_goal))

        if render and start_hold > 0:
            hold_until = time.time() + start_hold
            while time.time() < hold_until:
                env.render()
                time.sleep(0.03)

        # If eval_mass is given, override the default source/target cube mass.
        # We set it after reset because the environment may recreate/reset dynamics.
        if eval_mass is not None:
            set_object_mass(env, eval_mass)

        terminated = False
        truncated = False
        episode_return = 0.0
        step = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

            if render:
                env.render()
                time.sleep(render_sleep)

            step += 1

        if render and end_hold > 0:
            hold_until = time.time() + end_hold
            while time.time() < hold_until:
                env.render()
                time.sleep(0.03)

        episode_returns.append(episode_return)
        episode_lengths.append(step)
        final_achieved_goal = np.array(obs["achieved_goal"], dtype=np.float32)
        final_distance = float(np.linalg.norm(final_achieved_goal - desired_goal))
        object_move = float(np.linalg.norm(final_achieved_goal - start_achieved_goal))
        start_distances.append(start_distance)
        final_distances.append(final_distance)
        object_moves.append(object_move)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(
            f"Episode {episode:03d} | "
            f"return = {episode_return:.3f} | "
            f"steps = {step} | "
            f"start_dist = {start_distance:.3f} | "
            f"final_dist = {final_distance:.3f} | "
            f"object_move = {object_move:.3f}"
        )

    returns = np.array(episode_returns, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.float32)
    start_distances = np.array(start_distances, dtype=np.float32)
    final_distances = np.array(final_distances, dtype=np.float32)
    object_moves = np.array(object_moves, dtype=np.float32)

    mean_return = float(returns.mean())
    std_return = float(returns.std())
    min_return = float(returns.min())
    max_return = float(returns.max())
    mean_len = float(lengths.mean())

    success_rate = None
    if successes:
        success_rate = float(np.mean(successes))

    print("\n=== Evaluation summary ===")
    print(f"Experiment name: {experiment_name}")
    print(f"Model path: {model_path}")
    print(f"Environment type: {env_type}")
    print(f"Eval mass override: {eval_mass}")
    print(f"Seed: {seed}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"Mean return: {mean_return:.3f}")
    print(f"Std return:  {std_return:.3f}")
    print(f"Min return:  {min_return:.3f}")
    print(f"Max return:  {max_return:.3f}")
    print(f"Mean episode length: {mean_len:.2f}")
    print(f"Mean start distance: {start_distances.mean():.3f}")
    print(f"Mean final distance: {final_distances.mean():.3f}")
    print(f"Mean object movement: {object_moves.mean():.3f}")

    if success_rate is not None:
        print(f"Success rate: {success_rate:.2%}")

    if save_csv:
        append_results_to_csv(
            csv_path=csv_path,
            experiment_name=experiment_name,
            model_path=model_path,
            env_type=env_type,
            eval_mass=eval_mass,
            n_episodes=n_episodes,
            seed=seed,
            deterministic=deterministic,
            mean_return=mean_return,
            std_return=std_return,
            min_return=min_return,
            max_return=max_return,
            success_rate=success_rate,
        )

        print(f"\nSaved evaluation row to: {csv_path}")

    if render:
        input("\nPress Enter to close the PandaPush window...")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO or SAC on PandaPush-v3")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO/SAC model zip file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of eval episodes",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window",
    )

    parser.add_argument(
        "--render-sleep",
        type=float,
        default=0.03,
        help="Seconds to wait after each rendered environment step",
    )

    parser.add_argument(
        "--start-hold",
        type=float,
        default=0.0,
        help="Seconds to hold the initial rendered state before acting",
    )

    parser.add_argument(
        "--end-hold",
        type=float,
        default=0.0,
        help="Seconds to hold the final rendered state before closing",
    )

    parser.add_argument(
        "--env-type",
        type=str,
        default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on",
    )

    parser.add_argument(
        "--eval-mass",
        type=float,
        default=None,
        help=(
            "Optional manual object mass override for robustness tests. "
            "Example: --eval-mass 3.0"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for evaluation resets",
    )

    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Append evaluation summary to a CSV file",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/eval_results.csv",
        help="CSV file where evaluation summaries are stored",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="manual_eval",
        help="Name/label for this evaluation experiment",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        render_sleep=args.render_sleep,
        start_hold=args.start_hold,
        end_hold=args.end_hold,
        env_type=args.env_type,
        eval_mass=args.eval_mass,
        seed=args.seed,
        save_csv=args.save_csv,
        csv_path=args.csv_path,
        experiment_name=args.experiment_name,
    )
