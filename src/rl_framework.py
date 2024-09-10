import torch
import pyro
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cvae_scm import load_cvae_and_create_scm

# ------------------ Reward Shaping ------------------

def reward_shaping(Y, target_shape=1):
    """
    Reward shaping for the agent's goal of reaching the target shape.
    Returns a high positive reward for matching the target, with penalties for distance.
    """
    distance_to_target = torch.abs(Y[1] - target_shape).item()
    return 100 - distance_to_target * 10  # Reward decreases with distance to the target

# ------------------ Exploration Bonus ------------------

def exploration_bonus(agent, Z, discovered_latent_states):
    """
    Provide an exploration bonus if the agent discovers a new latent state.
    `discovered_latent_states` keeps track of previously discovered latent states.
    """
    Z_tuple = tuple(Z.squeeze().tolist())  # Convert latent state to a hashable tuple
    if Z_tuple not in discovered_latent_states:
        discovered_latent_states.add(Z_tuple)
        return 10.0  # Exploration bonus for discovering a new latent state
    return 0.0

# ------------------ Importance Weight Calculation ------------------

def importance_weight(action, likelihood_new, likelihood_old):
    """
    Compute the importance sampling weight given the new and old likelihoods.
    The weight represents how much we should upweight or downweight this action.
    This function now handles multiple elements by returning an average weight across all elements.
    """
    # Element-wise division and taking the mean across all elements
    weights = (likelihood_new / (likelihood_old + 1e-8)).detach()

    # Return the mean of the importance weights
    return weights.mean().item()

# ------------------ Policy Network Definition ------------------

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ------------------ Environment Definitions ------------------

class SimpleEnv:
    def __init__(self, scm=None):
        """
        If scm is None, the environment acts as non-causal (Vanilla).
        If scm is provided, it behaves as a causal environment.
        """
        self.scm = scm
        self.state = None
        self.non_causal = scm is None  # If SCM is None, treat as non-causal.
        self.reset()

    def reset(self):
        if self.non_causal:
            # Non-causal case, just random initial state from latent space
            self.state = torch.randn(1, self.scm.vae.z_dim).detach()
        else:
            # Causal case, use SCM to generate initial state
            (X, Y, Z), _ = self.scm()
            self.state = Z.detach()
        return self.state

    def step(self, action, discovered_latent_states):
        # Apply action, but clip to ensure action values stay reasonable
        new_Z = self.state + torch.clamp(torch.tensor(action), -0.1, 0.1)  # Small, clipped action perturbation

        if self.non_causal:
            # Non-causal: no SCM, we just generate a random reward based on the latent action
            reward = -torch.abs(new_Z.mean() - 0.5).item()  # Reward based on proximity to a mean value
        else:
            # Causal: Use SCM to generate new states based on modified Z
            cond_data = {"Z": new_Z}
            conditioned_model = pyro.condition(self.scm.model, data=cond_data)
            (X, Y, _), _ = conditioned_model(self.scm.init_noise)
            reward = reward_shaping(Y)

        # Exploration bonus for discovering new latent states
        reward += exploration_bonus(self, new_Z, discovered_latent_states)

        self.state = new_Z  # Update state
        return self.state, reward

# ------------------ RL Agent Definition ------------------

class RLAgent:
    def __init__(self, state_dim, action_dim, exploration_noise=0.1, entropy_coef=0.01, lr=0.001):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.exploration_noise = exploration_noise
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_mean = self.policy(state)
        action_dist = torch.distributions.Normal(action_mean, torch.ones_like(action_mean) * self.exploration_noise)
        action = action_dist.sample()

        # Add entropy regularization to encourage exploration
        log_prob = action_dist.log_prob(action).sum() + self.entropy_coef * action_dist.entropy().sum()
        return action.detach().numpy(), log_prob, action_dist

    def update(self, rewards, log_probs, importance_weights=None):
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        if importance_weights is not None:
            loss = -torch.stack(log_probs) * torch.FloatTensor(rewards) * torch.FloatTensor(importance_weights)
        else:
            loss = -torch.stack(log_probs) * torch.FloatTensor(rewards)
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ------------------ Training Function ------------------

def train_rl_agent(env, agent, num_episodes, discovered_latent_states=None, use_importance_sampling=False):
    if discovered_latent_states is None:
        discovered_latent_states = set()  # Set to track discovered latent states

    reward_history = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        log_probs = []
        rewards = []
        importance_weights = []

        for t in range(200):  # Run for 200 steps
            action, log_prob, action_dist = agent.select_action(state)
            next_state, reward = env.step(action, discovered_latent_states)

            log_probs.append(log_prob)
            rewards.append(reward)

            if use_importance_sampling:
                # Apply importance sampling to adjust for causal weights
                likelihood_new = action_dist.log_prob(torch.FloatTensor(action))
                likelihood_old = log_prob
                imp_weight = importance_weight(action, likelihood_new, likelihood_old)
                importance_weights.append(imp_weight)

            state = next_state
            total_reward += reward

        # Update the policy using the rewards and log probabilities
        if use_importance_sampling:
            agent.update(rewards, log_probs, importance_weights)
        else:
            agent.update(rewards, log_probs)
        reward_history.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return reward_history

# ------------------ Vanilla, SCM-Based, and Importance Sampling Agents ------------------

def compare_agents(scm, vae, num_episodes=300):
    discovered_latent_states = set()

    # Vanilla (Non-Causal) Agent
    vanilla_env = SimpleEnv()  # Non-causal environment
    vanilla_agent = RLAgent(state_dim=vae.z_dim, action_dim=vae.z_dim)

    # SCM-Based Causal Inference Agent
    scm_env = SimpleEnv(scm=scm)  # Causal environment with SCM
    causal_agent = RLAgent(state_dim=vae.z_dim, action_dim=vae.z_dim)

    # SCM-Based Importance Sampling Agent
    importance_sampling_env = SimpleEnv(scm=scm)  # Causal environment with SCM
    importance_sampling_agent = RLAgent(state_dim=vae.z_dim, action_dim=vae.z_dim)

    # Train Vanilla Agent
    print("Training Vanilla (Non-Causal) Agent...")
    vanilla_rewards = train_rl_agent(vanilla_env, vanilla_agent, num_episodes, discovered_latent_states)

    # Train SCM-Based Causal Inference Agent
    print("Training SCM-Based Causal Inference Agent (Interventional Reasoning)...")
    causal_rewards = train_rl_agent(scm_env, causal_agent, num_episodes, discovered_latent_states)

    # Train SCM-Based Importance Sampling Agent
    print("Training SCM-Based Importance Sampling Agent...")
    advanced_rewards = train_rl_agent(importance_sampling_env, importance_sampling_agent, num_episodes, discovered_latent_states, use_importance_sampling=True)

    # Plotting Comparison
    plt.figure(figsize=(14, 6))
    plt.plot(vanilla_rewards, label="Vanilla (Non-Causal) Agent")
    plt.plot(causal_rewards, label="SCM-Based Causal Inference Agent (Interventional Reasoning)")
    plt.plot(advanced_rewards, label="SCM-Based Importance Sampling Agent")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Performance Comparison of Different RL Agents")
    plt.legend()
    plt.show()

# ------------------ Main Execution ------------------

def main():
    # Load CVAE and create SCM
    cvae_path = 'path_to_your_trained_cvae.pth'
    context_dim = 10  # Adjust as needed
    item_dim = 20     # Adjust as needed
    z_dim = 5         # Adjust as needed
    vae, scm = load_cvae_and_create_scm(cvae_path, context_dim, item_dim, z_dim)

    # Run the comparison between different RL agents with causal and non-causal approaches
    compare_agents(scm, vae, num_episodes=300)

if __name__ == "__main__":
    main()