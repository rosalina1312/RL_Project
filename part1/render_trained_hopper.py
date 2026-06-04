"""Render the required from-scratch Actor-Critic policy on Hopper-v4."""

import time
import argparse

import gymnasium as gym
import mujoco
import numpy as np
import torch

from agent import Policy, Agent


def reset_env(env, seed: int):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env.reset(seed=seed)


def set_follow_camera(env):
    try:
        env.render()

        viewer = env.unwrapped.mujoco_renderer.viewer
        model = env.unwrapped.model
        torso_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso",
        )

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

    except Exception as error:
        print("Could not set tracking camera:", error)


def update_camera_position(env):
    try:
        viewer = env.unwrapped.mujoco_renderer.viewer
        torso_x = float(env.unwrapped.data.qpos[0])
        torso_z = float(env.unwrapped.data.qpos[1])

        viewer.cam.lookat[0] = torso_x
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = max(0.8, torso_z)

    except Exception as error:
        print("Could not update camera position:", error)


def main():
    parser = argparse.ArgumentParser(description="Render from-scratch Actor-Critic Hopper policy.")
    parser.add_argument("--model-path", type=str, default="best_actor_critic_policy.pth")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sleep-time", type=float, default=0.0)
    parser.add_argument("--sim-steps-per-render", type=int, default=3)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--init-sigma", type=float, default=0.5)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the policy mean. This is the default for the submitted Actor-Critic policy.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the learned stochastic policy instead of using the deterministic mean.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    env = gym.make("Hopper-v4", render_mode="human")

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(
        state_space,
        action_space,
        hidden_size=args.hidden_size,
        init_sigma=args.init_sigma,
    )
    policy.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
    policy.eval()

    agent = Agent(policy, algorithm="actor_critic", baseline=None)

    print("Loaded trained policy from:", args.model_path)
    print("Duration:", args.duration)
    print("Sim steps per render:", args.sim_steps_per_render)
    print("Action scale:", args.action_scale)
    print("Sleep time:", args.sleep_time)
    use_deterministic = args.deterministic or not args.stochastic
    print("Action mode:", "deterministic mean" if use_deterministic else "stochastic sample")

    start_time = time.time()
    episode = 0

    while time.time() - start_time < args.duration:
        state, info = reset_env(env, args.seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        start_x = float(env.unwrapped.data.qpos[0])

        set_follow_camera(env)

        while not done and time.time() - start_time < args.duration:
            for _ in range(args.sim_steps_per_render):
                if done:
                    break

                with torch.no_grad():
                    action, _ = agent.get_action(state, evaluation=use_deterministic)

                action_np = args.action_scale * action.detach().cpu().numpy()
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

                state, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                total_reward += reward
                steps += 1

            update_camera_position(env)
            env.render()
            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

        end_x = float(env.unwrapped.data.qpos[0])
        print(
            f"Episode {episode} | "
            f"reward = {total_reward:.2f} | "
            f"steps = {steps} | "
            f"x_delta = {end_x - start_x:.3f}"
        )

        episode += 1

    env.close()


if __name__ == "__main__":
    main()
