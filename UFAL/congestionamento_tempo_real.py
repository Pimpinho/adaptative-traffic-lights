import traci
import matplotlib.pyplot as plt

# Inicializa SUMO com TraCI
sumoCmd = ["sumo", "-c", "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg"]  # ou sumo-gui
traci.start(sumoCmd)

# Configura gráfico ao vivo
plt.ion()
fig, ax = plt.subplots()

times = []
waiting_times = []

step = 0
while step < 3600:  # simulação de 3600 segundos
    traci.simulationStep()

    # Lista de lanes
    lanes = traci.lane.getIDList()
    
    # Calcula tempo total de espera em todas as lanes
    total_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)

    # Armazena dados
    times.append(step)
    waiting_times.append(total_waiting)

    # Atualiza gráfico ao vivo
    ax.clear()
    ax.plot(times, waiting_times, label='Tempo de Espera Total')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Tempo de Espera (s)')
    ax.set_title('Veículos parados na simulação')
    ax.legend()
    plt.pause(0.001)  # pequeno delay para atualizar gráfico

    step += 1

traci.close()

plt.ioff()
plt.show()
