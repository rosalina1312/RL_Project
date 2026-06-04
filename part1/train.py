"""Systematic Part 1 experiments: REINFORCE and Actor-Critic on Hopper-v4."""

import argparse
import json
import os
import sys
import time
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from agent import Policy, Agent


def set_seed(seed: int) -> None:
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


def get_runtime_versions():
    return {
        "python_version": sys.version.split()[0],
        "gymnasium_version": gym.__version__,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def print_runtime_versions():
    versions = get_runtime_versions()
    version_text = ", ".join(f"{name}={value}" for name, value in versions.items())
    print("Runtime versions:", version_text)


def save_run_metadata(args, output_path):
    metadata = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "args": vars(args),
        "runtime_versions": get_runtime_versions(),
    }

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def moving_average(values, window=10):
    values = np.array(values, dtype=np.float32)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def make_experiment_name(algorithm, baseline):
    if algorithm == "reinforce":
        if baseline is None:
            return "reinforce_no_baseline"
        return f"reinforce_baseline_{baseline}"

    if algorithm == "actor_critic":
        return "actor_critic"

    return algorithm


def train_agent(
    algorithm="reinforce",
    baseline=None,
    n_episodes=300,
    render=False,
    seed=42,
    hidden_size=256,
    init_sigma=0.5,
    init_method="normal",
    learning_rate=1e-3,
    gamma=0.99,
    value_coef=0.5,
    entropy_coef=0.0,
    max_grad_norm=None,
    normalize_advantages=False,
    normalize_observations=False,
    squash_action_mean=False,
    actor_critic_update="returns",
    save_best_path=None,
    save_final_path=None,
    best_metric="reward",
    print_every=1,
):
    set_seed(seed)

    if render:
        env = gym.make("Hopper-v4", render_mode="human")
    else:
        env = gym.make("Hopper-v4")

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    print("\n======================================")
    print("Experiment:", make_experiment_name(algorithm, baseline))
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("State dimension:", state_space)
    print("Action dimension:", action_space)
    print("Seed:", seed)
    print_runtime_versions()
    print("======================================")

    policy = Policy(
        state_space,
        action_space,
        hidden_size=hidden_size,
        init_sigma=init_sigma,
        init_method=init_method,
        normalize_observations=normalize_observations,
        squash_action_mean=squash_action_mean,
    )
    agent = Agent(
        policy,
        algorithm=algorithm,
        baseline=baseline,
        learning_rate=learning_rate,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        normalize_advantages=normalize_advantages,
        actor_critic_update=actor_critic_update,
    )

    episode_rewards = []
    episode_x_deltas = []
    best_score = -float("inf")
    best_reward = None
    best_x_delta = None
    start_time = time.time()

    for episode in range(n_episodes):
        state, info = reset_env(env, seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        start_x = float(env.unwrapped.data.qpos[0])

        while not done:
            action, action_log_prob = agent.get_action(state)

            action_np = action.detach().cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()

        loss = agent.update_policy()
        end_x = float(env.unwrapped.data.qpos[0])
        x_delta = end_x - start_x
        episode_rewards.append(total_reward)
        episode_x_deltas.append(x_delta)

        if best_metric == "x_delta":
            score = x_delta
        else:
            score = total_reward

        if save_best_path is not None and algorithm == "actor_critic" and score > best_score:
            best_score = score
            best_reward = total_reward
            best_x_delta = x_delta
            torch.save(policy.state_dict(), save_best_path)

        if print_every > 0 and ((episode + 1) % print_every == 0 or episode == 0):
            print(
                f"Episode {episode + 1:03d}/{n_episodes} | "
                f"Reward: {total_reward:8.2f} | "
                f"x_delta: {x_delta:7.3f} | "
                f"Steps: {steps:3d} | "
                f"Loss: {loss:8.4f}"
            )

    if save_final_path is not None and algorithm == "actor_critic":
        torch.save(policy.state_dict(), save_final_path)

    elapsed_time = time.time() - start_time
    env.close()

    final_avg_10 = float(np.mean(episode_rewards[-10:]))
    final_avg_50 = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards))
    avg_reward = float(np.mean(episode_rewards))
    final_avg_x_delta_50 = float(np.mean(episode_x_deltas[-50:])) if len(episode_x_deltas) >= 50 else float(np.mean(episode_x_deltas))

    print("\nTraining finished.")
    print("Algorithm:", algorithm)
    print("Baseline:", baseline)
    print("Total time:", elapsed_time, "seconds")
    print("Average reward:", avg_reward)
    print("Final average reward last 10:", final_avg_10)
    print("Final average reward last 50:", final_avg_50)
    print("Final average x_delta last 50:", final_avg_x_delta_50)
    if save_best_path is not None and algorithm == "actor_critic":
        print("Saved best Actor-Critic policy:", save_best_path)
        print("Best saved reward:", best_reward)
        print("Best saved x_delta:", best_x_delta)

    return {
        "episode_rewards": episode_rewards,
        "episode_x_deltas": episode_x_deltas,
        "elapsed_time": elapsed_time,
        "avg_reward": avg_reward,
        "final_avg_10": final_avg_10,
        "final_avg_50": final_avg_50,
        "final_avg_x_delta_50": final_avg_x_delta_50,
        "best_reward": best_reward,
        "best_x_delta": best_x_delta,
    }


def plot_learning_curves(results_df, experiment_names, output_path):
    window = 10

    plt.figure(figsize=(11, 6))

    for name in experiment_names:
        rewards = results_df[name].values
        ma_rewards = moving_average(rewards, window)

        if len(ma_rewards) == len(rewards):
            x = np.arange(1, len(ma_rewards) + 1)
        else:
            x = np.arange(window, len(rewards) + 1)

        plt.plot(x, ma_rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Episode reward, moving average over 10 episodes")
    plt.title("Part 1: REINFORCE Baselines vs Actor-Critic on Hopper-v4")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_actor_critic_policy(policy, episodes=3, seed=10_000, deterministic=True):
    env = gym.make("Hopper-v4")
    returns = []
    x_deltas = []
    lengths = []

    was_training = policy.training
    policy.eval()

    for episode in range(episodes):
        state, info = reset_env(env, seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        start_x = float(env.unwrapped.data.qpos[0])

        while not (terminated or truncated):
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float()
                normal_dist, _ = policy(state_tensor)
                if deterministic:
                    action = normal_dist.mean
                else:
                    action = normal_dist.sample()

            action_np = np.clip(
                action.detach().cpu().numpy(),
                env.action_space.low,
                env.action_space.high,
            )
            state, reward, terminated, truncated, info = env.step(action_np)
            total_reward += float(reward)
            steps += 1

        end_x = float(env.unwrapped.data.qpos[0])
        returns.append(total_reward)
        x_deltas.append(end_x - start_x)
        lengths.append(steps)

    env.close()
    if was_training:
        policy.train()

    return {
        "mean_return": float(np.mean(returns)),
        "mean_x_delta": float(np.mean(x_deltas)),
        "mean_length": float(np.mean(lengths)),
        "max_x_delta": float(np.max(x_deltas)),
    }


def train_actor_critic_rollout(
    total_timesteps=300_000,
    seed=42,
    hidden_size=256,
    init_sigma=-0.5,
    init_method="normal",
    learning_rate=3e-4,
    gamma=0.99,
    value_coef=0.5,
    entropy_coef=0.001,
    max_grad_norm=0.5,
    normalize_observations=False,
    squash_action_mean=False,
    rollout_steps=32,
    n_envs=8,
    eval_every=25_000,
    eval_episodes=5,
    save_best_path="best_actor_critic_policy.pth",
    save_final_path="final_actor_critic_policy.pth",
    print_every=10,
):
    set_seed(seed)

    envs = [gym.make("Hopper-v4") for _ in range(n_envs)]
    state_space = envs[0].observation_space.shape[0]
    action_space = envs[0].action_space.shape[0]
    action_low = envs[0].action_space.low
    action_high = envs[0].action_space.high

    policy = Policy(
        state_space,
        action_space,
        hidden_size=hidden_size,
        init_sigma=init_sigma,
        init_method=init_method,
        normalize_observations=normalize_observations,
        squash_action_mean=squash_action_mean,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    states = []
    episode_start_x = []
    current_rewards = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=np.int32)

    for env_index, env in enumerate(envs):
        state, info = reset_env(env, seed + env_index)
        states.append(state)
        episode_start_x.append(float(env.unwrapped.data.qpos[0]))

    states = np.asarray(states, dtype=np.float32)
    episode_start_x = np.asarray(episode_start_x, dtype=np.float32)

    episode_rewards = []
    episode_x_deltas = []
    episode_lengths = []
    total_steps = 0
    update = 0
    next_eval = eval_every
    best_score = -float("inf")
    best_eval = None
    start_time = time.time()

    print("\n======================================")
    print("Experiment: actor_critic_rollout")
    print("State dimension:", state_space)
    print("Action dimension:", action_space)
    print("Seed:", seed)
    print("Total timesteps:", total_timesteps)
    print("Parallel environments:", n_envs)
    print("Rollout steps:", rollout_steps)
    print_runtime_versions()
    print("======================================")

    while total_steps < total_timesteps:
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []
        rollout_entropies = []

        for _ in range(rollout_steps):
            policy.update_observation_statistics(states)
            state_tensor = torch.from_numpy(states).float()
            normal_dist, values = policy(state_tensor)
            actions = normal_dist.sample()
            log_probs = normal_dist.log_prob(actions).sum(dim=-1)
            entropies = normal_dist.entropy().sum(dim=-1)

            actions_np = actions.detach().cpu().numpy()
            clipped_actions = np.clip(actions_np, action_low, action_high)

            next_states = []
            rewards = np.zeros(n_envs, dtype=np.float32)
            dones = np.zeros(n_envs, dtype=np.float32)

            for env_index, env in enumerate(envs):
                next_state, reward, terminated, truncated, info = env.step(clipped_actions[env_index])
                done = terminated or truncated

                rewards[env_index] = float(reward)
                dones[env_index] = float(done)
                current_rewards[env_index] += float(reward)
                current_lengths[env_index] += 1

                if done:
                    end_x = float(env.unwrapped.data.qpos[0])
                    episode_rewards.append(float(current_rewards[env_index]))
                    episode_x_deltas.append(float(end_x - episode_start_x[env_index]))
                    episode_lengths.append(int(current_lengths[env_index]))

                    next_state, info = reset_env(env, seed + total_steps + env_index + 1)
                    episode_start_x[env_index] = float(env.unwrapped.data.qpos[0])
                    current_rewards[env_index] = 0.0
                    current_lengths[env_index] = 0

                next_states.append(next_state)

            rollout_log_probs.append(log_probs)
            rollout_values.append(values.squeeze(-1))
            rollout_rewards.append(torch.from_numpy(rewards))
            rollout_dones.append(torch.from_numpy(dones))
            rollout_entropies.append(entropies)

            states = np.asarray(next_states, dtype=np.float32)
            total_steps += n_envs

            if total_steps >= total_timesteps:
                break

        with torch.no_grad():
            next_state_tensor = torch.from_numpy(states).float()
            _, next_values = policy(next_state_tensor)
            returns = next_values.squeeze(-1)

        computed_returns = []
        for step in reversed(range(len(rollout_rewards))):
            returns = rollout_rewards[step] + gamma * returns * (1.0 - rollout_dones[step])
            computed_returns.insert(0, returns)

        log_probs_tensor = torch.stack(rollout_log_probs)
        values_tensor = torch.stack(rollout_values)
        returns_tensor = torch.stack(computed_returns).detach()
        entropies_tensor = torch.stack(rollout_entropies)

        advantages = returns_tensor - values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        critic_loss = F.mse_loss(values_tensor, returns_tensor)
        entropy_loss = entropies_tensor.mean()
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        update += 1

        should_eval = total_steps >= next_eval or total_steps >= total_timesteps
        if should_eval:
            eval_result = evaluate_actor_critic_policy(
                policy,
                episodes=eval_episodes,
                seed=20_000 + update,
                deterministic=True,
            )
            score = eval_result["mean_x_delta"]
            if score > best_score:
                best_score = score
                best_eval = eval_result
                torch.save(policy.state_dict(), save_best_path)

            next_eval += eval_every

        if print_every > 0 and (update % print_every == 0 or should_eval):
            recent_rewards = episode_rewards[-20:] if episode_rewards else [0.0]
            recent_x = episode_x_deltas[-20:] if episode_x_deltas else [0.0]
            print(
                f"Update {update:04d} | steps={total_steps:7d} | "
                f"recent_reward={np.mean(recent_rewards):8.2f} | "
                f"recent_x_delta={np.mean(recent_x):6.3f} | "
                f"loss={loss.item():8.4f}"
            )
            if should_eval and best_eval is not None:
                print(
                    "Best deterministic eval so far | "
                    f"mean_return={best_eval['mean_return']:.2f} | "
                    f"mean_x_delta={best_eval['mean_x_delta']:.3f} | "
                    f"max_x_delta={best_eval['max_x_delta']:.3f}"
                )

    torch.save(policy.state_dict(), save_final_path)

    for env in envs:
        env.close()

    elapsed_time = time.time() - start_time
    final_rewards = episode_rewards[-50:] if episode_rewards else [0.0]
    final_x = episode_x_deltas[-50:] if episode_x_deltas else [0.0]

    print("\nRollout Actor-Critic training finished.")
    print("Total time:", elapsed_time, "seconds")
    print("Episodes completed:", len(episode_rewards))
    print("Final average reward last 50:", float(np.mean(final_rewards)))
    print("Final average x_delta last 50:", float(np.mean(final_x)))
    print("Best saved eval:", best_eval)
    print("Saved best Actor-Critic policy:", save_best_path)
    print("Saved final Actor-Critic policy:", save_final_path)

    return {
        "episode_rewards": episode_rewards,
        "episode_x_deltas": episode_x_deltas,
        "elapsed_time": elapsed_time,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "final_avg_10": float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0,
        "final_avg_50": float(np.mean(final_rewards)),
        "final_avg_x_delta_50": float(np.mean(final_x)),
        "best_reward": None if best_eval is None else best_eval["mean_return"],
        "best_x_delta": None if best_eval is None else best_eval["mean_x_delta"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train required Part 1 Hopper policies.")
    parser.add_argument("--algorithm", choices=["all", "reinforce", "actor_critic"], default="all")
    parser.add_argument("--actor-critic-mode", choices=["episode", "rollout"], default="episode")
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--init-sigma", type=float, default=0.5)
    parser.add_argument("--init-method", choices=["normal", "orthogonal"], default="normal")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--squash-action-mean", action="store_true")
    parser.add_argument("--actor-critic-update", choices=["returns", "td"], default="returns")
    parser.add_argument("--best-metric", choices=["reward", "x_delta"], default="reward")
    parser.add_argument("--save-best-path", default="best_actor_critic_policy.pth")
    parser.add_argument("--save-final-path", default="final_actor_critic_policy.pth")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--output-prefix", default="part1")
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    n_episodes = args.episodes
    seed = args.seed
    runtime_versions = get_runtime_versions()
    metadata_path = f"{args.output_prefix}_run_metadata.json"
    save_run_metadata(args, metadata_path)
    print("Saved run metadata:", metadata_path)

    if args.algorithm == "actor_critic" and args.actor_critic_mode == "rollout":
        result = train_actor_critic_rollout(
            total_timesteps=args.timesteps,
            seed=seed,
            hidden_size=args.hidden_size,
            init_sigma=args.init_sigma,
            init_method=args.init_method,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            normalize_observations=args.normalize_observations,
            squash_action_mean=args.squash_action_mean,
            rollout_steps=args.rollout_steps,
            n_envs=args.n_envs,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            save_best_path=args.save_best_path,
            save_final_path=args.save_final_path,
            print_every=args.print_every,
        )

        summary_df = pd.DataFrame([{
            "experiment": "actor_critic_rollout",
            "algorithm": "actor_critic",
            "baseline": None,
            "episodes": len(result["episode_rewards"]),
            "timesteps": args.timesteps,
            "seed": seed,
            "avg_reward": result["avg_reward"],
            "final_avg_10": result["final_avg_10"],
            "final_avg_50": result["final_avg_50"],
            "final_avg_x_delta_50": result["final_avg_x_delta_50"],
            "best_reward": result["best_reward"],
            "best_x_delta": result["best_x_delta"],
            "elapsed_time_sec": result["elapsed_time"],
            **runtime_versions,
        }])
        summary_path = f"{args.output_prefix}_summary.csv"
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            existing_df = existing_df[existing_df["algorithm"] != "actor_critic"]
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)
        print("\nSaved CSV:", summary_path)
        return

    if args.algorithm == "all":
        experiments = [
            {"algorithm": "reinforce", "baseline": None},
            {"algorithm": "reinforce", "baseline": 10.0},
            {"algorithm": "reinforce", "baseline": 20.0},
            {"algorithm": "reinforce", "baseline": 50.0},
            {"algorithm": "actor_critic", "baseline": None},
        ]
    else:
        experiments = [
            {"algorithm": args.algorithm, "baseline": args.baseline},
        ]

    all_rewards = {}
    summary_rows = []

    for exp in experiments:
        algorithm = exp["algorithm"]
        baseline = exp["baseline"]
        name = make_experiment_name(algorithm, baseline)

        result = train_agent(
            algorithm=algorithm,
            baseline=baseline,
            n_episodes=n_episodes,
            render=False,
            seed=seed,
            hidden_size=args.hidden_size,
            init_sigma=args.init_sigma,
            init_method=args.init_method,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            normalize_advantages=args.normalize_advantages,
            normalize_observations=args.normalize_observations,
            squash_action_mean=args.squash_action_mean,
            actor_critic_update=args.actor_critic_update,
            save_best_path=args.save_best_path if algorithm == "actor_critic" else None,
            save_final_path=args.save_final_path if algorithm == "actor_critic" else None,
            best_metric=args.best_metric,
            print_every=args.print_every,
        )

        all_rewards[name] = result["episode_rewards"]
        all_rewards[f"{name}_x_delta"] = result["episode_x_deltas"]

        summary_rows.append({
            "experiment": name,
            "algorithm": algorithm,
            "baseline": baseline,
            "episodes": n_episodes,
            "seed": seed,
            "avg_reward": result["avg_reward"],
            "final_avg_10": result["final_avg_10"],
            "final_avg_50": result["final_avg_50"],
            "final_avg_x_delta_50": result["final_avg_x_delta_50"],
            "best_reward": result["best_reward"],
            "best_x_delta": result["best_x_delta"],
            "elapsed_time_sec": result["elapsed_time"],
            **runtime_versions,
        })

    results_df = pd.DataFrame({
        "episode": np.arange(1, n_episodes + 1)
    })

    for name, rewards in all_rewards.items():
        results_df[name] = rewards

    summary_df = pd.DataFrame(summary_rows)

    results_path = f"{args.output_prefix}_results.csv"
    summary_path = f"{args.output_prefix}_summary.csv"
    plot_path = f"{args.output_prefix}_comparison_learning_curve.png"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_learning_curves(
        results_df=results_df,
        experiment_names=[name for name in all_rewards.keys() if not name.endswith("_x_delta")],
        output_path=plot_path,
    )

    print("\n========== Part 1 Summary ==========")
    print(summary_df)

    print("\nSaved CSV:", results_path)
    print("Saved CSV:", summary_path)
    print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()
