from dataclasses import dataclass
import random
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tyro


@dataclass
class Args:
    env_id: str = "CartPole-v1"
    """gym environment id"""
    lr: float = 2.5e-4
    """learning rate"""
    gamma: float = 0.9
    """discount factor"""
    num_episodes: int = 1000
    """number of episodes"""
    max_step: int = 500
    """max_step"""
    seed: int = 42
    """seed number"""


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.layers(state)


class REINFORCE:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=2.5e-4,
        gamma=0.9,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else self.device
        )

        self.gamma = gamma
        self.action_dim = action_dim
        self.policy = Policy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """select action using e-greedy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # unsqueeze because state from env has shape (state_dim,) - 1D array but neural network needs (batch_size, state_dim)
        # tensor([x1, x2, x3, x4]) -> tensor([[x1, x2, x3, x4]])
        action_probs = self.policy(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        self.log_probs.append(log_prob)

        return action.item()

    def calculate_returns(self, rewards):
        returns = deque()
        R = 0

        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.appendleft(R)

        return returns

    def update_policy(self):
        if not self.rewards:
            return

        returns = self.calculate_returns(self.rewards)
        returns = torch.FloatTensor(returns).to(self.device)

        # normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() - 1e8)

        # calucate policy loss: -log pi(a|s) * G_t
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # perform gradient descent
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

    def store_reward(self, reward):
        self.rewards.append(reward)


def train(args: Args):
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = REINFORCE(state_dim, action_dim, lr=args.lr, gamma=args.gamma)

    # training metrics
    episode_rewards = []
    running_rewards = deque(maxlen=100)

    print(f"Training REINFORCE on {args.env_id}")
    print(f"running on {agent.device}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Policy network: {state_dim} -> 128 -> 128 -> {action_dim}")
    print("-" * 50)

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        episode_reward: float = 0.0

        for step in range(args.max_step):
            action = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)

            agent.store_reward(reward)
            episode_reward += reward

            state = next_state

            if done or truncated:
                break

        agent.update_policy()

        # track metrics
        episode_rewards.append(episode_reward)
        running_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(running_rewards)
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {episode_reward:6.2f} | "
                f"Avg Reward: {avg_reward:6.2f}"
            )
    env.close()
    return agent, episode_rewards


def plot_training_progress(rewards, window=100):
    """Plot training progress."""
    plt.figure(figsize=(12, 4))

    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    # Plot smoothed rewards
    plt.subplot(1, 2, 2)
    smoothed = [
        np.mean(rewards[max(0, i - window) : i + 1]) for i in range(len(rewards))
    ]
    plt.plot(smoothed, linewidth=2)
    plt.title(f"Smoothed Rewards (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = tyro.cli(Args)
    agent, training_rewards = train(args)

    plot_training_progress(training_rewards)

    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "training_rewards": training_rewards,
            "hyperparameters": {
                "lr": 1e-3,
                "gamma": 0.99,
                "episodes": len(training_rewards),
            },
        },
        "reinforce_model.pth",
    )
    print("\nModel saved as 'reinforce_model.pth'")

    # Show final performance
    final_avg = np.mean(training_rewards[-100:])
    print(f"Final 100-episode average: {final_avg:.2f}")
