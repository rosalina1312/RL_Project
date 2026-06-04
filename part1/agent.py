import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0

    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


class Policy(torch.nn.Module):
    def __init__(
        self,
        state_space,
        action_space,
        hidden_size=256,
        init_sigma=0.5,
        init_method="normal",
        normalize_observations=False,
        squash_action_mean=False,
    ):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden_size
        self.init_method = init_method
        self.tanh = torch.nn.Tanh()

        self.register_buffer("obs_mean", torch.zeros(state_space))
        self.register_buffer("obs_var", torch.ones(state_space))
        self.register_buffer("obs_count", torch.tensor(1e-4))
        self.register_buffer(
            "obs_norm_enabled",
            torch.tensor(1.0 if normalize_observations else 0.0),
        )
        self.register_buffer(
            "squash_mean_enabled",
            torch.tensor(1.0 if squash_action_mean else 0.0),
        )

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        self.sigma_activation = F.softplus
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        if self.init_method == "orthogonal":
            gain = torch.nn.init.calculate_gain("tanh")

            for layer in [
                self.fc1_actor,
                self.fc2_actor,
                self.fc1_critic,
                self.fc2_critic,
            ]:
                torch.nn.init.orthogonal_(layer.weight, gain=gain)
                torch.nn.init.zeros_(layer.bias)

            torch.nn.init.orthogonal_(self.fc3_actor_mean.weight, gain=0.01)
            torch.nn.init.zeros_(self.fc3_actor_mean.bias)
            torch.nn.init.orthogonal_(self.fc3_critic_value.weight, gain=1.0)
            torch.nn.init.zeros_(self.fc3_critic_value.bias)
            return

        if self.init_method == "normal":
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                    torch.nn.init.zeros_(m.bias)
            return

        raise ValueError("Unknown init_method. Use 'normal' or 'orthogonal'.")

    def update_observation_statistics(self, observations):
        if self.obs_norm_enabled.item() == 0.0:
            return

        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.obs_mean.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(obs.shape[0]), device=self.obs_mean.device)

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count

        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.obs_count * batch_count / total_count
        new_var = m_2 / total_count

        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(torch.clamp(new_var, min=1e-6))
        self.obs_count.copy_(total_count)

    def forward(self, x):
        if self.obs_norm_enabled.item() == 1.0:
            x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
            x = torch.clamp(x, -10.0, 10.0)

        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)
        if self.squash_mean_enabled.item() == 1.0:
            action_mean = torch.tanh(action_mean)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(
        self,
        policy,
        device="cpu",
        algorithm="reinforce",
        baseline=None,
        learning_rate=1e-3,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=None,
        normalize_advantages=False,
        actor_critic_update="returns",
    ):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.algorithm = algorithm
        self.baseline = baseline
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.actor_critic_update = actor_critic_update

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        if self.algorithm == "reinforce":
            returns = discount_rewards(rewards, self.gamma)

            if self.baseline is not None:
                advantages = returns - self.baseline
            else:
                advantages = returns

            policy_loss = -(action_log_probs * advantages.detach()).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        elif self.algorithm == "actor_critic":
            normal_dist, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

            if self.actor_critic_update == "td":
                with torch.no_grad():
                    _, next_state_values = self.policy(next_states)
                    next_state_values = next_state_values.squeeze(-1)
                    targets = rewards + self.gamma * next_state_values * (1 - done)
            else:
                targets = discount_rewards(rewards, self.gamma)

            advantages = targets - state_values

            if self.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(action_log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(state_values, targets.detach())

            total_loss = actor_loss + self.value_coef * critic_loss

            if self.entropy_coef > 0.0:
                entropies = normal_dist.entropy().sum(dim=-1)
                total_loss = total_loss - self.entropy_coef * entropies.mean()

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            return total_loss.item()

        else:
            raise ValueError("Unknown algorithm. Use 'reinforce' or 'actor_critic'.")

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:
            return normal_dist.mean, None

        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
