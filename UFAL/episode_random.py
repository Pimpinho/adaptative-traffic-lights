# run_one_episode.py
import random
from sumo_env_norm import SUMOEnv

if __name__ == "__main__":
    SUMO_BINARY = "sumo"   # ou "sumo-gui" se quiser ver
    SUMO_CFG = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg"

    env = SUMOEnv(
        sumo_binary=SUMO_BINARY,
        sumo_cfg=SUMO_CFG,
        tl_ids=("tl1", "tl2", "tl3"),
        # lanes_by_tl pode ser omitido, já está default no env
        step_length=1.0,
        control_interval=5
    )

    # número de decisões do agente no episódio (5s cada -> 100*5 = 500s)
    NUM_DECISIONS = 100

    state = env.reset()
    print("Estado inicial (len={}):".format(len(state)))
    print(state)

    episode_reward = 0.0

    for step in range(NUM_DECISIONS):
        # por enquanto: agente aleatório
        action = random.choice([0, 1, 2])

        next_state, reward, done, info = env.step(
            action, episode=0, step_idx=step
        )

        episode_reward += reward

        print(f"Step {step:03d} | action={action} | sim_time={info['sim_time']:.1f} "
              f"| reward={reward:.2f} | phases={info['phases']}")

        state = next_state

        # se em algum momento você quiser encerrar quando não tiver mais veículos:
        # if traci.simulation.getMinExpectedNumber() <= 0:
        #     break

    env.close()

    print("\n=== Episódio finalizado ===")
    print(f"Decisões do agente: {step+1}")
    print(f"Retorno total do episódio (soma dos rewards): {episode_reward:.2f}")
    print("Veja o arquivo 'sumo_env_log.csv' para métricas detalhadas.")
