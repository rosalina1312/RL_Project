import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        mass_range=(0.5, 4.0),
        mode="none",
        adr_initial_range=(0.8, 1.2),
        adr_min_limit=0.5,
        adr_max_limit=4.0,
        adr_step=0.2,
        adr_success_threshold=0.7,
        adr_window_size=20,
        verbose=False,
        seed=None,
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range
        self.verbose = verbose

        self.mass_min_limit, self.mass_max_limit = mass_range
        self.adr_mass_min, self.adr_mass_max = adr_initial_range
        self.adr_min_limit = adr_min_limit
        self.adr_max_limit = adr_max_limit
        self.adr_step = adr_step
        self.adr_success_threshold = adr_success_threshold
        self.adr_window_size = adr_window_size

        self.recent_successes = []
        self.current_mass = None
        self.last_sample_type = "none"
        self.mass_min = self.mass_min_limit
        self.mass_max = self.mass_max_limit
        self.rng = np.random.default_rng(seed)

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "none"
            return None

        if self.mode == "udr":
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
            self.last_sample_type = "uniform"
            return self.rng.uniform(self.mass_min, self.mass_max)

        if self.mode == "adr":
            self.mass_min = self.adr_mass_min
            self.mass_max = self.adr_mass_max
            self.last_sample_type = "adaptive"
            return self.rng.uniform(self.adr_mass_min, self.adr_mass_max)

        raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented.")

    def _set_object_mass(self, mass):
        sim = self.env.unwrapped.task.sim
        object_body_id = sim._bodies_idx["object"]

        sim.physics_client.changeDynamics(
            bodyUniqueId=object_body_id,
            linkIndex=-1,
            mass=float(mass),
        )

    def _update_adr_range(self):
        if len(self.recent_successes) < self.adr_window_size:
            return

        success_rate = np.mean(self.recent_successes[-self.adr_window_size:])

        if success_rate >= self.adr_success_threshold:
            old_min = self.adr_mass_min
            old_max = self.adr_mass_max

            self.adr_mass_min = max(
                self.adr_min_limit,
                self.adr_mass_min - self.adr_step,
            )
            self.adr_mass_max = min(
                self.adr_max_limit,
                self.adr_mass_max + self.adr_step,
            )

            changed = old_min != self.adr_mass_min or old_max != self.adr_mass_max
            if changed or self.verbose:
                print(
                    f"[ADR] success_rate={success_rate:.2f} | "
                    f"range expanded: [{old_min:.2f}, {old_max:.2f}] "
                    f"-> [{self.adr_mass_min:.2f}, {self.adr_mass_max:.2f}]"
                )

            self.recent_successes = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if self.mode == "adr" and done:
            if isinstance(info, dict) and "is_success" in info:
                self.recent_successes.append(float(info["is_success"]))
                self._update_adr_range()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        new_mass = self._sample_mass()
        obs, info = self.env.reset(**kwargs)

        if new_mass is not None:
            self._set_object_mass(new_mass)
            self.current_mass = float(new_mass)

            if self.verbose:
                print(
                    f"[{self.mode}] mass={self.current_mass:.2f} "
                    f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                    f"type={self.last_sample_type}"
                )

        return obs, info
