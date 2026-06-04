"""Evaluate the Part 1 required from-scratch Actor-Critic Hopper policy."""

import argparse

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy


def reset_env(env, seed: int):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env.reset(seed=seed)


def evaluate(model_path, episodes, seed, hidden_size, init_sigma, deterministic):
    env = gym.make("Hopper-v4")

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space, hidden_size=hidden_size, init_sigma=init_sigma)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    policy.eval()

    agent = Agent(policy, algorithm="actor_critic", baseline=None)

    returns = []
    lengths = []
    x_deltas = []

    for episode in range(episodes):
        state, info = reset_env(env, seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        start_x = float(env.unwrapped.data.qpos[0])

        while not (terminated or truncated):
            with torch.no_grad():
                action, _ = agent.get_action(state, evaluation=deterministic)

            action_np = np.clip(
                action.detach().cpu().numpy(),
                env.action_space.low,
                env.action_space.high,
            )

            state, reward, terminated, truncated, info = env.step(action_np)
            total_reward += float(reward)
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)
        end_x = float(env.unwrapped.data.qpos[0])
        x_delta = end_x - start_x
        x_deltas.append(x_delta)
        print(
            f"Episode {episode + 1:03d} | "
            f"return={total_reward:.2f} | steps={steps} | x_delta={x_delta:.3f}"
        )

    env.close()

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)
    x_deltas = np.array(x_deltas, dtype=np.float32)

    print("\n=== Required Actor-Critic evaluation ===")
    print("Model:", model_path)
    print("Episodes:", episodes)
    print(f"Mean return: {returns.mean():.2f}")
    print(f"Std return:  {returns.std():.2f}")
    print(f"Min return:  {returns.min():.2f}")
    print(f"Max return:  {returns.max():.2f}")
    print(f"Mean length: {lengths.mean():.2f}")
    print(f"Mean x_delta: {x_deltas.mean():.3f}")
    print(f"Max x_delta:  {x_deltas.max():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="best_actor_critic_policy.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--init-sigma", type=float, default=0.5)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of using the deterministic mean.",
    )
    args = parser.parse_args()

    evaluate(
        args.model_path,
        args.episodes,
        args.seed,
        args.hidden_size,
        args.init_sigma,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
