"""DQN Agent for OpenAI Gym Environments
============================================
Train a DQN agent on any Gym environment with discrete actions.

The agent uses a Deep Q-Network (DQN) with:
  - Experience replay buffer
  - Target network for stable training
  - Epsilon-greedy exploration with decay

Requirements:
  pip install gym[box2d] torch numpy

Usage:
  python spaceship.py --train                           # Train on LunarLander (default)
  python spaceship.py --train --env CartPole-v1         # Train on CartPole
  python spaceship.py --train --env MountainCar-v0      # Train on MountainCar
  python spaceship.py --test                            # Watch trained agent
  python spaceship.py --train --test                    # Train then watch
"""

import argparse
import random
import os
from collections import deque, namedtuple

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Hyperparameters ───────────────────────────────────────────────
EPISODES        = 400       # Number of training episodes
MAX_STEPS       = 1000      # Max steps per episode
BATCH_SIZE      = 64        # Mini-batch size for training
GAMMA           = 0.99      # Discount factor
LR              = 5e-4      # Learning rate
MEMORY_SIZE     = 100_000   # Replay buffer capacity
TAU             = 1e-3      # Soft-update rate for target network
EPSILON_START   = 1.0       # Initial exploration rate
EPSILON_END     = 0.01      # Minimum exploration rate
EPSILON_DECAY   = 0.995     # Decay rate per episode
UPDATE_EVERY    = 4         # Learn every N steps
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))

def get_model_path(env_name):
    """Return model save path based on environment name."""
    safe_name = env_name.replace("-", "_").lower()
    return os.path.join(SCRIPT_DIR, f"dqn_{safe_name}.pth")

# ─── Experience tuple ──────────────────────────────────────────────
Experience = namedtuple("Experience",
                        ["state", "action", "reward", "next_state", "done"])


# ═══════════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)

        states  = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones   = torch.FloatTensor(np.array([e.done for e in experiences]).astype(np.uint8)).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# ═══════════════════════════════════════════════════════════════════
#  Q-Network
# ═══════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    """Neural network that approximates Q(s, a)."""

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════
#  DQN Agent
# ═══════════════════════════════════════════════════════════════════
class DQNAgent:
    """Agent that learns using Deep Q-Learning."""

    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        self.step_count  = 0

        # Q-Networks (online + target)
        self.qnetwork_local  = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory    = ReplayBuffer(MEMORY_SIZE)

    def act(self, state, epsilon=0.0):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        """Store experience and learn if enough samples are available."""
        self.memory.push(state, action, reward, next_state, done)
        self.step_count += 1

        if self.step_count % UPDATE_EVERY == 0 and len(self.memory) >= BATCH_SIZE:
            self._learn()

    def _learn(self):
        """Update Q-network using a batch of experiences."""
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Current Q-values for chosen actions
        q_current = self.qnetwork_local(states).gather(1, actions)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            q_target_next = self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0]
        q_target = rewards + GAMMA * q_target_next * (1 - dones)

        # Compute loss and update
        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update target network: θ_target ← τ·θ_local + (1-τ)·θ_target
        for target_param, local_param in zip(
                self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path, weights_only=True))
        self.qnetwork_local.eval()
        print(f"Model loaded from {path}")


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════
def train(agent, env, env_name, num_episodes):
    """Train the DQN agent on the given Gym environment."""
    model_path = get_model_path(env_name)
    scores = []
    epsilon = EPSILON_START
    best_avg = -float('inf')

    print("=" * 60)
    print(f"  Training DQN Agent on {env_name}")
    print("=" * 60)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0

        for t in range(MAX_STEPS):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Print progress every 50 episodes
        avg_score = np.mean(scores[-100:])
        if episode % 50 == 0:
            print(f"  Episode {episode:4d} | "
                  f"Avg Score: {avg_score:7.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Last: {score:7.1f}")

        # Save best model
        if avg_score > best_avg and episode >= 100:
            best_avg = avg_score
            agent.save(model_path)

        # Solved! (average reward >= 200 over last 100 episodes)
        if avg_score >= 200.0 and episode >= 100:
            print(f"\n  *** SOLVED in {episode} episodes! "
                  f"Avg Score: {avg_score:.1f} ***\n")
            agent.save(model_path)
            break

    print("Training complete.\n")
    return scores


# ═══════════════════════════════════════════════════════════════════
#  Testing / Demonstration
# ═══════════════════════════════════════════════════════════════════
def test(agent, env_name, num_episodes=5):
    """Watch the trained agent play the environment."""
    env = gym.make(env_name, render_mode="human")

    print("=" * 60)
    print(f"  Watching Trained Agent on {env_name}")
    print("=" * 60)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0

        for t in range(MAX_STEPS):
            action = agent.act(state, epsilon=0.0)  # No exploration
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward

            if terminated or truncated:
                break

        print(f"  Episode {episode}: Score = {score:.1f}")

    env.close()
    print("\nDemonstration complete.")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on any Gym environment")
    parser.add_argument('--train', action='store_true',
                        help='Train the agent')
    parser.add_argument('--test', action='store_true',
                        help='Test/watch the trained agent')
    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='Gym environment name (default: LunarLander-v2)')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                        help=f'Number of training episodes (default: {EPISODES})')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    if not args.train and not args.test:
        print("Please specify --train, --test, or both.")
        print("Example: python spaceship.py --train --env CartPole-v1")
        exit(1)

    env_name = args.env

    # Auto-detect state and action sizes from the environment
    temp_env = gym.make(env_name)
    state_size  = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.n
    temp_env.close()

    print(f"Environment: {env_name}")
    print(f"State size:  {state_size}")
    print(f"Action size: {action_size}\n")

    # Create agent
    agent = DQNAgent(state_size, action_size)
    model_path = get_model_path(env_name)

    # ── Train ──
    if args.train:
        env = gym.make(env_name)
        scores = train(agent, env, env_name, args.episodes)
        env.close()

    # ── Test ──
    if args.test:
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"Warning: No saved model found at {model_path}")
            print("Running with untrained agent.\n")
        test(agent, env_name)
