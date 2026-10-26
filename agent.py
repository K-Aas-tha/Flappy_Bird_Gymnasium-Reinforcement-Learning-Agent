import flappy_bird_gymnasium
import gymnasium as gym
import torch
import numpy as np
from dqn import DQN
from ex_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim
import os
import argparse


if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.env_id = params["env_id"]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.rewards_threshhold = params["rewards_threshhold"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.network_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]
        self.replay_memory_size = params["replay_memory_size"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        with gym.make(
            self.env_id, render_mode="human" if render else None
        ) as env:
            num_states = env.observation_space.shape[0]
            num_actions = env.action_space.n
            policy_dqn = DQN(num_states, num_actions).to(device)

            if is_training:
                memory = ReplayMemory(self.replay_memory_size)
                epsilon = self.epsilon_init
                target_dqn = DQN(num_states, num_actions).to(device)
                target_dqn.load_state_dict(policy_dqn.state_dict())
                steps = 0
                self.optimizer = optim.Adam(
                    policy_dqn.parameters(), lr=self.alpha
                )
                best_rewards = float("-inf")
            else:
                policy_dqn.load_state_dict(
                    torch.load(self.MODEL_FILE, map_location=device)
                )
                policy_dqn.eval()

            for episode in itertools.count():
                states, _ = env.reset()
                states = torch.tensor(states, dtype=torch.float, device=device)
                episode_rewards = 0
                terminated = False

                while not terminated and episode_rewards < self.rewards_threshhold:
                    if is_training and np.random.random() < epsilon:
                        actions = env.action_space.sample()
                        actions = torch.tensor(
                            actions, dtype=torch.long, device=device
                        )
                    else:
                        with torch.no_grad():
                            actions = (
                                policy_dqn(states.unsqueeze(dim=0))
                                .squeeze(dim=0)
                                .argmax()
                            )

                    next_states, rewards, terminated, truncated, _ = env.step(
                        actions.item()
                    )
                    episode_done = terminated or truncated
                    rewards = torch.tensor(
                        rewards, dtype=torch.float, device=device
                    )
                    next_states = torch.tensor(
                        next_states, dtype=torch.float, device=device
                    )

                    if is_training:
                        memory.append(
                            (states, actions, next_states, rewards, terminated)
                        )
                        steps += 1

                    states = next_states
                    episode_rewards += rewards.item()
                    if episode_done:
                        break

                epsilon_message = f"{epsilon:.4f}" if is_training else "n/a"
                print(
                    f"Episode {episode + 1}: total reward={episode_rewards}, "
                    f"epsilon={epsilon_message}"
                )
                if is_training:
                    epsilon = max(
                        epsilon * self.epsilon_decay, self.epsilon_min
                    )

                    if episode_rewards > best_rewards:
                        log_msg = (
                            f"best rewards= {episode_rewards} "
                            f"for ep {episode + 1}"
                        )
                        with open(self.LOG_FILE, "a") as f:
                            f.write(log_msg + "\n")

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_rewards = episode_rewards

                if is_training and len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(dim=1)
        ).squeeze(dim=1)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("hyperparameters", help=" ")
    parser.add_argument("--train", help="training_mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)
    if args.train:
        dql.run()
    else:
        dql.run(is_training=False, render=True)