from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import tyro
import random
from collections import deque
# from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_id: str = "CartPole-v1"
    """gym environment id"""
    lr: float = 2.5e-4
    """learning rate"""
    gamma: float = 0.99
    """discount factor"""
    num_episodes: int = 500_000
    """number of episodes"""
    max_step: int = 500
    """max_step"""
    seed: int = 42
    """seed number"""
    buffer_limit: int = 50_000
    """max limit for relay buffer"""
    batch_size: int = 128
    """batch size of sample from relay memory"""
    epsilon_max: float = 1.0
    """max epsilon"""
    epsilon_min: float = 0.05
    """min epsilon"""
    epsilon_decay: float = 0.5
    """fraction of adjust epsilon"""
    target_update: int = 500
    """the timesteps it takes to update the target network"""


class RelayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=128):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr,
        gamma,
        epsilon_max,
        epsilon_min,
        epsilon_decay,
        buffer_limit,
        batch_size,
        target_update,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else self.device
        )

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.q_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.memory = RelayBuffer(buffer_limit)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn from batch of experiences"""
        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        # update target network periodically
        if self.t_step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, state, episode):
        """choose action using epsilon-greedy"""
        self.epsilon = self.epsilon_min + (
            self.epsilon_max - self.epsilon_min
        ) * np.exp(-self.epsilon_decay * episode)
        if random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state)
        return np.argmax(q_values.cpu().data.numpy())

    def learn(self, experiences):
        """update Q-network using batch of exp"""
        states, actions, rewards, next_states, dones = experiences

        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))

        # next q value from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(args: Args, log_dir="runs/dqn"):
    env = gym.make(args.env_id)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size,
        action_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_max=args.epsilon_max,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_limit=args.buffer_limit,
        batch_size=args.batch_size,
        target_update=args.target_update,
    )
    print(f"running on {agent.device}")

    # writer = SummaryWriter(log_dir)
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    episode_rewards = []
    episode_losses = []

    for episode in range(args.num_episodes):
        state, _ = env.reset()

        episode_reward = 0
        losses = []

        for step in range(args.max_step):
            action = agent.act(state, episode)
            next_state, reward, done, truncated, _ = env.step(action)

            done = done or truncated

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

        # log metrics
        # writer.add_scalar("Episode/Reward", episode_reward, episode)
        # writer.add_scalar("Episode/Epsilon", agent.epsilon, episode)
        # writer.add_scalar("Episode/Steps", step + 1, episode)

        if episode_losses:
            avg_loss = np.mean(losses)
            episode_losses.append(avg_loss)
            # writer.add_scalar("Episode/Loss", avg_loss, episode)

        if episode % 100 == 0:
            avg_score = np.mean(episode_rewards)
            print(
                f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}"
            )

    # writer.close()
    env.close()
    return agent


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(f"Training DQN agent on {args.env_id}")
    train(args)

    print("\nTo view training logs, run: tensorboard --logdir=runs")
