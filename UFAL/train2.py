# train_dqn.py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from sumo_env_norm import SUMOEnv  # importa seu ambiente pronto


# ============================
#   DQN Network
# ============================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================
#   Replay Buffer
# ============================

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state e next_state: np.array
        self.buffer.append((
            state.astype(np.float32),
            action,
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.stack(states))
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states))
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================
#   Funções Auxiliares
# ============================

def select_action(state, policy_net, epsilon, action_dim, device):
    """
    Epsilon-greedy:
      - com prob epsilon: ação aleatória
      - senão: argmax Q(state, a)
    """
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = policy_net(state_t)
            action = q_values.argmax(dim=1).item()
            return action


def optimize_model(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None  # ainda não tem amostras suficientes

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Q(s,a) atual
    q_values = policy_net(states).gather(1, actions)

    # Q alvo usando target_net
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        target = rewards + gamma * (1.0 - dones) * next_q_values

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item()


# ============================
#   Loop de Treino
# ============================

def train_dqn(
    num_episodes=50,
    max_sim_time=3600,      # segundos de simulação por episódio
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=40,
    target_update_interval=1000   # steps de treino
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Cria ambiente
    env = SUMOEnv(
        sumo_binary="sumo",  # "sumo" ou "sumo-gui"
        # sumo_cfg já está no SUMOEnv por padrão, ou você passa aqui
        step_length=1.0,
        control_interval=5
    )

    # Descobre dimensão do estado
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = 3  # 3 ações: 0=tl1, 1=tl2, 2=tl3

    print(f"state_dim = {state_dim}, action_dim = {action_dim}")

    # Redes
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=100_000)

    global_step = 0

    # Treino por episódios
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        losses = []

        # epsilon decaindo por episódio
        frac = min(episode / epsilon_decay_episodes, 1.0)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        print(f"\n=== EPISODE {episode+1}/{num_episodes} | epsilon={epsilon:.3f} ===")

        step_idx = 0
        while True:
            # Seleciona ação
            action = select_action(state, policy_net, epsilon, action_dim, device)

            # Executa no ambiente
            next_state, reward, done, info = env.step(action, episode=episode, step_idx=step_idx)
            sim_time = info["sim_time"]

            # Armazena na memória
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            step_idx += 1
            global_step += 1

            # Atualiza DQN
            loss = optimize_model(policy_net, target_net, optimizer,
                                  replay_buffer, batch_size, gamma, device)
            if loss is not None:
                losses.append(loss)

            # Atualiza target_net periodicamente
            if global_step % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"[global_step={global_step}] Target network updated.")

            # critério de término do episódio
            if sim_time >= max_sim_time:
                break

        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Episode {episode+1}: total_reward={episode_reward:.2f} | avg_loss={avg_loss:.4f}")

    # Fecha ambiente ao final
    env.close()
    # Salva modelo treinado
    torch.save(policy_net.state_dict(), "dqn_traffic_lights.pth")
    print("\nTreino finalizado. Modelo salvo em dqn_traffic_lights.pth")


if __name__ == "__main__":
    train_dqn(
        num_episodes=10,      # começa pequeno para testar
        max_sim_time=3600,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=8,
        target_update_interval=1000
    )
