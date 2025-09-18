import matplotlib.pyplot as plt
import numpy as np

# --- Seus Resultados ---
fixed_time_wait = 238588.00
dqn_wait_time = 74870.00
percentage_improvement = ((fixed_time_wait - dqn_wait_time) / fixed_time_wait) * 100

# --- Configuração do Gráfico ---
labels = ['Modelo com DQN', 'Tempo Fixo']
wait_times = [dqn_wait_time, fixed_time_wait]
colors = ['blue', 'orange']

fig, ax = plt.subplots(figsize=(8, 6))

# Cria as barras
bars = ax.bar(labels, wait_times, color=colors)

# Adiciona os rótulos de dados em cima das barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,.2f}', va='bottom', ha='center', fontsize=12, fontweight='bold')

# --- Estilização do Gráfico ---
ax.set_ylabel('Tempo de Espera Total (segundos)')
ax.set_title('Comparação do Tempo de Espera Total\nEntre o Modelo DQN e o Sistema de Tempo Fixo')
# Formata o eixo Y para ser mais legível
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adiciona o texto da melhoria percentual no topo
plt.suptitle(f'Melhora Percentual do Modelo DQN: {percentage_improvement:.2f}%', fontsize=14, fontweight='bold', y=0.98)

# Salva e exibe o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("comparacao_resultados.png")
plt.show()