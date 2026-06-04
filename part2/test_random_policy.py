"""Test a random policy on PandaPush-v3."""

import gymnasium as gym
import panda_gym  # required so PandaPush-v3 is registered


def main():
    render = False

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type="target",
        reward_type="dense",
    )

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    n_episodes = 5

    for _ in range(n_episodes):
        done = False
        state, info = env.reset()

        while not done:
            action = env.action_space.sample()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if render:
                env.render()

    env.close()


if __name__ == "__main__":
    main()
