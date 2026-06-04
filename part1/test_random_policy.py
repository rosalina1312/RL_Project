"""Test a random policy on the Gym Hopper environment visually for ~15 seconds."""

import time

import gymnasium as gym
import mujoco


def main():
    render = True
    duration_seconds = 15

    env = gym.make(
        "Hopper-v4",
        render_mode="human" if render else "rgb_array",
    )

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    model = env.unwrapped.model

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]

    print("Body names:", body_names)
    print("Body masses:", model.body_mass)
    print("Number of bodies:", model.nbody)
    print("Number of DoFs nv:", model.nv)
    print("Body DoF numbers:", model.body_dofnum)
    print("Number of actuators nu:", model.nu)

    start_time = time.time()
    episode = 0

    while time.time() - start_time < duration_seconds:
        done = False
        step = 0
        state, info = env.reset()

        print("\nEpisode:", episode)
        print("Initial state shape:", state.shape)

        while not done and time.time() - start_time < duration_seconds:
            action = env.action_space.sample()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if step < 5:
                print("\nStep:", step)
                print("Action:", action)
                print("Reward:", reward)
                print("Terminated:", terminated)
                print("Truncated:", truncated)

            env.render()
            time.sleep(0.03)

            step += 1

        print("Episode ended after", step, "steps")
        episode += 1

    input("\n15 seconds finished. Press Enter to close the Hopper window...")
    env.close()


if __name__ == "__main__":
    main()