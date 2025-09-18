# -*- coding: utf-8 -*-
"""
Script para medir o tempo de espera total em uma simulação SUMO
com semáforos de tempo fixo (estáticos).
Este código serve como linha de base (baseline) para comparar com
o desempenho do agente de Reinforcement Learning.
"""

import os
import sys

# --- CONFIGURAÇÃO DO AMBIENTE SUMO ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Por favor, declare a variável de ambiente 'SUMO_HOME'")

import traci

# --- CONSTANTES DE CONFIGURAÇÃO ---

# ATENÇÃO: Use "sumo" para a medição mais rápida e precisa.
# A GUI consome recursos e pode alterar ligeiramente os resultados.
SUMO_BINARY = "sumo"

# ATENÇÃO: Substitua pelo mesmo caminho de arquivo .sumocfg usado no treinamento.
SUMO_CFG_PATH = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg"

# Lista dos cruzamentos que você quer medir
JUNCTION_IDS = ['6022602047', '2621508821', '795820931']

# Duração da simulação (deve ser a mesma do treinamento para uma comparação justa)
SIMULATION_STEPS = 3600

# ==============================================================================
# SCRIPT PRINCIPAL DE MEDIÇÃO
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando simulação de linha de base (tempo fixo)...")
    
    # Monta o comando para iniciar o SUMO
    sumo_cmd = [
        SUMO_BINARY, 
        "-c", SUMO_CFG_PATH,
        "--no-warnings", "true",
        "--waiting-time-memory", "10000" # Aumenta a memória para medições precisas
    ]
    
    # Inicia a simulação
    traci.start(sumo_cmd)
    
    # --- Inicialização ---
    step = 0
    total_cumulative_wait_time = 0.0
    
    # Coleta todas as faixas de acesso aos cruzamentos de interesse uma única vez
    lanes_to_monitor = []
    for j_id in JUNCTION_IDS:
        lanes_to_monitor.extend(traci.trafficlight.getControlledLanes(j_id))
    # Remove duplicatas caso haja faixas compartilhadas
    lanes_to_monitor = sorted(list(set(lanes_to_monitor)))
    
    print(f"Monitorando {len(lanes_to_monitor)} faixas nos {len(JUNCTION_IDS)} cruzamentos.")

    # --- Loop da Simulação ---
    while step < SIMULATION_STEPS:
        # Avança um segundo na simulação
        traci.simulationStep()
        
        # Calcula o tempo de espera somado de todas as faixas monitoradas NESTE passo
        wait_time_this_step = sum(traci.lane.getWaitingTime(lane) for lane in lanes_to_monitor)
        
        # Adiciona o valor deste passo ao total acumulado
        total_cumulative_wait_time += wait_time_this_step

        # Imprime o progresso a cada 500 segundos para sabermos que está rodando
        if step % 500 == 0:
            print(f"  Passo: {step}/{SIMULATION_STEPS} | Tempo de Espera Acumulado: {total_cumulative_wait_time:.2f} s")
            
        step += 1
        
    # Fecha a conexão com o SUMO
    traci.close()
    
    # --- Resultado Final ---
    print("\nSimulação de linha de base finalizada!")
    print("==========================================================")
    print(f"TEMPO DE ESPERA TOTAL (TEMPO FIXO): {total_cumulative_wait_time:.2f} segundos")
    print("==========================================================")