# -*- coding: utf-8 -*-
"""
Implementação de um Deep Q-Network (DQN) para Controle de Semáforos no SUMO.
Algoritmo adaptado do artigo "Real Time Traffic Light Timing Optimisation Using Reinforcement Learning"
para a rede e semáforos especificados pelo usuário.
"""

import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from collections import deque
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DO AMBIENTE SUMO ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Por favor, declare a variável de ambiente 'SUMO_HOME'")

import traci

# --- CONSTANTES E HIPERPARÂMETROS ---

# Configuração da Simulação
SUMO_BINARY = "sumo"  # Use "sumo" para treinamento rápido sem interface gráfica
# ATENÇÃO: Substitua pelo caminho correto do seu arquivo de configuração .sumocfg
SUMO_CFG_PATH = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg" 
JUNCTION_IDS = ['6022602047', '2621508821', '795820931']
EPISODES = 5  # Número de simulações de treinamento
SIMULATION_STEPS = 3600 # Duração de cada simulação em segundos

# Hiperparâmetros do DQN (baseado no artigo)
ACTION_SIZE = 2  # Ação 0: Manter fase, Ação 1: Mudar fase
MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.95  # Fator de desconto
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Parâmetros de Controle do Semáforo
DECISION_INTERVAL = 10  # Agente toma uma decisão a cada 10 segundos
YELLOW_TIME = 4         # Duração da fase amarela em segundos

# Dicionário para armazenar informações dinâmicas dos cruzamentos
junction_info = {}

# ==============================================================================
# CLASSE DO AGENTE DEEP Q-NETWORK
# ==============================================================================
class DQNAgent:
    """
    Classe do Agente DQN conforme descrito no artigo[cite: 93].
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = self._build_model()

    def _build_model(self):
        """
        Constrói a rede neural com 3 camadas densas e ativação ReLU[cite: 94].
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE)) # Usa MSE Loss e Adam Optimizer [cite: 97]
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Armazena uma transição (estado, ação, recompensa, etc.) na memória[cite: 104].
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy[cite: 96].
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """
        Treina a rede com um lote de amostras da memória[cite: 105].
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([transition[0] for transition in minibatch]).reshape(batch_size, self.state_size)
        next_states = np.array([transition[3] for transition in minibatch]).reshape(batch_size, self.state_size)
        
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                target = reward + GAMMA * np.amax(q_next[i])
            else:
                target = reward
            q_current[i][action] = target

        self.model.fit(states, q_current, epochs=1, verbose=0)

    def adapt_epsilon(self):
        """
        Aplica o decaimento ao epsilon para reduzir a exploração[cite: 106].
        """
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save(self, name):
        self.model.save_weights(name)

# ==============================================================================
# FUNÇÕES DE INTERAÇÃO COM O SUMO
# ==============================================================================

def initialize_junction_info():
    """
    Coleta informações dos semáforos (faixas de acesso, fases verdes/amarelas)
    e as armazena no dicionário global `junction_info`.
    """
    print("Inicializando informações dos cruzamentos...")
    for j_id in JUNCTION_IDS:
        incoming_lanes = sorted(list(set(traci.trafficlight.getControlledLanes(j_id))))
        
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(j_id)[0]
        green_phases_indices = [i for i, phase in enumerate(logic.phases) if 'g' in phase.state.lower() and 'y' not in phase.state.lower()]
        
        yellow_phase_map = {}
        for i, green_idx in enumerate(green_phases_indices):
            next_phase_idx_in_cycle = (i + 1) % len(green_phases_indices)
            next_green_phase_actual_idx = green_phases_indices[next_phase_idx_in_cycle]
            # Assumimos que a fase amarela é a que vem imediatamente antes da *próxima* fase verde.
            # Se a próxima fase verde é a 0, a amarela é a última do ciclo.
            yellow_idx = next_green_phase_actual_idx - 1 if next_green_phase_actual_idx > 0 else len(logic.phases) - 1
            yellow_phase_map[green_idx] = yellow_idx

        junction_info[j_id] = {
            'incoming_lanes': incoming_lanes,
            'state_size': len(incoming_lanes),
            'green_phases': green_phases_indices,
            'yellow_phase_map': yellow_phase_map,
            'current_green_phase_idx': green_phases_indices[0],
            'time_since_decision': 0,
            'last_action': 0
        }
        print(f"  - Cruzamento '{j_id}': Estado com {len(incoming_lanes)} faixas. Fases verdes encontradas: {green_phases_indices}")
        if len(green_phases_indices) < 2:
            print(f"    -> ATENÇÃO: Cruzamento '{j_id}' tem menos de 2 fases verdes. A ação de 'mudar' terá efeito limitado.")


def get_state(junction_id):
    """
    Retorna o estado do cruzamento (número de veículos nas faixas de acesso).
    """
    lanes = junction_info[junction_id]['incoming_lanes']
    state = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
    return np.array(state).reshape(1, junction_info[junction_id]['state_size'])

def get_reward(junction_id):
    """
    Retorna a recompensa (negativo do tempo de espera total nas faixas).
    """
    lanes = junction_info[junction_id]['incoming_lanes']
    return -sum(traci.lane.getWaitingTime(lane) for lane in lanes)

# ==============================================================================
# LOOP PRINCIPAL DE TREINAMENTO
# ==============================================================================
if __name__ == "__main__":
    agents = {}
    total_wait_time_history = []

    for episode in range(EPISODES):
        traci.start([SUMO_BINARY, "-c", SUMO_CFG_PATH, "--no-warnings", "true", "--waiting-time-memory", "10000"])
        
        if episode == 0:
            initialize_junction_info()
            for j_id in JUNCTION_IDS:
                agents[j_id] = DQNAgent(junction_info[j_id]['state_size'], ACTION_SIZE)
                
        current_states = {j_id: get_state(j_id) for j_id in JUNCTION_IDS}
        
        step = 0
        episode_total_wait_time = 0
        
        while step < SIMULATION_STEPS:
            # Ação e aprendizado ocorrem em intervalos
            if int(step) % DECISION_INTERVAL == 0:
                for j_id in JUNCTION_IDS:
                    # 1. ESCOLHER E APLICAR AÇÃO
                    action = agents[j_id].act(current_states[j_id])
                    junction_info[j_id]['last_action'] = action
                    
                    if action == 1: # Ação: Mudar para a próxima fase
                        # Só muda se houver mais de uma fase verde para escolher
                        if len(junction_info[j_id]['green_phases']) > 1:
                            current_green = junction_info[j_id]['current_green_phase_idx']
                            
                            # Transição para fase amarela
                            yellow_phase = junction_info[j_id]['yellow_phase_map'][current_green]
                            traci.trafficlight.setPhase(j_id, yellow_phase)
                            for _ in range(YELLOW_TIME):
                                traci.simulationStep()
                                step += 1
                                episode_total_wait_time += sum(traci.lane.getWaitingTime(l) for l in junction_info[j_id]['incoming_lanes'])

                            # Encontra o índice da próxima fase verde no ciclo
                            current_cycle_idx = junction_info[j_id]['green_phases'].index(current_green)
                            next_cycle_idx = (current_cycle_idx + 1) % len(junction_info[j_id]['green_phases'])
                            next_green = junction_info[j_id]['green_phases'][next_cycle_idx]
                            
                            # Transição para próxima fase verde
                            traci.trafficlight.setPhase(j_id, next_green)
                            junction_info[j_id]['current_green_phase_idx'] = next_green
                    
                # Deixa a simulação rodar pelo restante do intervalo
                interval_remaining = DECISION_INTERVAL - (step % DECISION_INTERVAL) if (step % DECISION_INTERVAL != 0) else DECISION_INTERVAL
                # Se a fase amarela já rodou, ajusta o tempo restante
                if action == 1 and len(junction_info[j_id]['green_phases']) > 1:
                    interval_remaining -= YELLOW_TIME

                for _ in range(interval_remaining):
                    traci.simulationStep()
                    step += 1
                    episode_total_wait_time += sum(traci.lane.getWaitingTime(l) for l in junction_info[j_id]['incoming_lanes'])

                # 2. COLETAR DADOS E TREINAR
                for j_id in JUNCTION_IDS:
                    reward = get_reward(j_id)
                    next_state = get_state(j_id)
                    done = step >= SIMULATION_STEPS
                    
                    agents[j_id].remember(current_states[j_id], junction_info[j_id]['last_action'], reward, next_state, done)
                    current_states[j_id] = next_state
                    agents[j_id].replay(BATCH_SIZE)
            else:
                traci.simulationStep()
                step += 1
                episode_total_wait_time += sum(traci.lane.getWaitingTime(l) for l in junction_info[j_id]['incoming_lanes'])


        traci.close()
        for agent in agents.values():
            agent.adapt_epsilon()
            
        total_wait_time_history.append(episode_total_wait_time)
        print(f"Episódio: {episode+1}/{EPISODES} | Tempo de Espera Total: {episode_total_wait_time:.2f} | Epsilon: {agents[JUNCTION_IDS[0]].epsilon:.4f}")

    print("\nTreinamento concluído!")

    # --- PLOTAR GRÁFICO DE DESEMPENHO (conforme Figura 1 do artigo) ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(EPISODES), total_wait_time_history)
    plt.title('Desempenho do Agente - Tempo de Espera por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Tempo de Espera Total Acumulado (s)')
    plt.grid(True)
    plt.savefig("desempenho_treinamento.png")
    plt.show()