import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Ambiente simulado com tempo de fase definido
class TrafficEnv:
    def __init__(self):
        self.green_durations = {0: 10, 1: 15}  # duração do verde: 0=NS, 1=EW
        self.reset()

    def reset(self):
        self.vehicles = np.random.randint(0, 10, size=4)  # [N, S, E, W]
        self.current_phase = 0
        self.phase_timer = self.green_durations[self.current_phase]
        self.time_step = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate((self.vehicles, [self.current_phase, self.phase_timer]))

    def step(self, action):
        reward = 0

        # Se a fase já está em andamento, não troca até terminar
        if self.phase_timer > 0:
            self.phase_timer -= 1
        else:
            # Troca de fase com base na ação do agente
            if action in self.green_durations:
                self.current_phase = action
                self.phase_timer = self.green_durations[action] - 1  # já conta 1 step aqui

        # Simula passagem de veículos na direção do verde
        passed = [0, 0, 0, 0]
        if self.current_phase == 0:  # NS verde
            passed[0] = min(self.vehicles[0], np.random.randint(1, 3))
            passed[1] = min(self.vehicles[1], np.random.randint(1, 3))
        elif self.current_phase == 1:  # EW verde
            passed[2] = min(self.vehicles[2], np.random.randint(1, 3))
            passed[3] = min(self.vehicles[3], np.random.randint(1, 3))

        self.vehicles -= passed
        self.vehicles = np.maximum(self.vehicles, 0)

        # Novos carros chegam
        self.vehicles += np.random.randint(0, 3, size=4)

        self.time_step += 1
        done = self.time_step >= 100  # episódio termina após 100 steps
        next_state = self._get_state()

        reward = -np.sum(self.vehicles)  # quanto menos veículos, melhor

        return next_state, reward, done

# Deep Q-Network (mesma estrutura)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Treinamento com suporte a tempos de fase
def train():
    num_episodes = 200
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    batch_size = 32
    memory = ReplayBuffer(10000)

    env = TrafficEnv()
    state_size = len(env.reset())
    action_size = 2  # 0 = NS verde, 1 = EW verde

    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    rewards_all = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                current_q = model(states).gather(1, actions)
                next_q = model(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + gamma * next_q * (1 - dones)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        rewards_all.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episódio {episode}, Recompensa: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # Gráfico de desempenho
    plt.plot(rewards_all)
    plt.title("Recompensa por episódio")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa acumulada")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train()
