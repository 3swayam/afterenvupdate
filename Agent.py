import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque  # ✅ Optimized memory management
import time



class ReplayMemory:
    def __init__(self, capacity):

        self.capacity = capacity
        self.memory = deque(maxlen=capacity)  # ✅ Uses deque for efficiency

    def store(self, state, action, next_state, reward, done):
        """
        Stores experience tuples for training.
        """
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size, device):
        """
        Retrieves a batch of random experiences for training.
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.int64).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.bool).to(device),
        )

    def __len__(self):
        return len(self.memory)


class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, lr=1e-3):
        """
        Defines the deep Q-network architecture.
        """
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def save_model(self, filename):
        """ Saves the policy network model. """
        torch.save(self.policy_net.state_dict(), filename)
        print(f"✅ Model saved as {filename}")

    def forward(self, x):
        """
        Forward pass through the DQN network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))


class DQNAgent:
    def __init__(self, device, state_size, action_size, discount=0.99, eps_max=1.0, eps_min=0.01, eps_decay=0.995, memory_capacity=25000, lr=1e-3, train_mode=True):

        self.device = device
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(capacity=memory_capacity)
        self.train_mode = train_mode

    def save_model(self, filename):

        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):

        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy_net.eval()

    def update_target_net(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # def select_action(self, state):

    #     if random.random() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
    #     with torch.no_grad():
    #         return torch.argmax(self.policy_net(state_tensor)).item()
        
    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()

    def learn(self, batch_size):
        """
        Trains the DQN agent using experience replay.
        """
        if len(self.memory) < batch_size:
            return

        start_time = time.time()  # ⏳ Track Learning Time

        states, actions, next_states, rewards, dones = self.memory.sample(batch_size, self.device)

        q_pred = self.policy_net(states).gather(1, actions.view(-1, 1))
        q_target = self.target_net(next_states).max(dim=1).values
        q_target[dones] = 0.0

        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()

        elapsed_time = time.time() - start_time  # 🔍 Track Execution Time

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))