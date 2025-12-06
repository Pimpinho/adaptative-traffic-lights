"""
dqn_multi_agent_sumo.py
Multi-Agent DQN for SUMO (1 agent per traffic light).

Assumptions from user:
- Agents: tl1, tl2, tl3
- Lanes:
  tl1 -> ['E5', 'dOrigem', 'inter1Origem']
  tl2 -> ['-E5', 'E1']
  tl3 -> ['E4', 'saidaufal', '-E1']
- Simulation length: 3600 seconds, step-length=1.0
- Using PyTorch

Usage examples:
  Train:
    python dqn_multi_agent_sumo.py --mode train
  Eval (no training):
    python dqn_multi_agent_sumo.py --mode eval --checkpoint checkpoint.pth
"""

import os
import random
import argparse
from collections import deque, namedtuple
import numpy as np
import time
import math

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# TraCI
try:
    import traci
except Exception as e:
    raise RuntimeError("traci import failed. Ensure SUMO is installed and python bindings are available.") from e

# -----------------------
# User paths & settings
# -----------------------
# Use the NET_FILE you provided
NET_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalNetwork.net.xml"

# SUMOCFG - change if needed. If you have a local .sumocfg file, set path here.
# I will default to a file name 'ufalConfig.sumocfg' in the same folder as NET_FILE.
SUMOCFG = os.path.join(os.path.dirname(NET_FILE), "ufalConfig.sumocfg")

# SUMO binary (it must be in PATH); adjust if needed
SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")  # or "sumo-gui"

# Simulation settings
EPISODE_STEPS = 3600  # seconds as you requested
STEP_LENGTH = 1.0

# Agent / environment config (defaults you allowed)
DEFAULT_MIN_GREEN = 5
DEFAULT_MAX_GREEN = 60
DEFAULT_MIN_RED = 5
DEFAULT_MAX_RED = 60
DEFAULT_YELLOW = 4

# TL definitions (from your messages)
TL_CONFIGS = {
    "tl1": {
        "lanes": ["E5", "dOrigem", "inter1Origem"],
        "yellow": 4,
        "phases": [42, 4, 42, 4]
    },
    "tl2": {
        "lanes": ["-E5", "E1"],
        "yellow": 4,
        "phases": [48, 4, 48, 4]
    },
    "tl3": {
        "lanes": ["E4", "saidaufal", "-E1"],
        "yellow": 4,
        "phases": [38, 4, 38, 4]
    }
}

# RL hyperparams (sensible defaults; tweak as needed)
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_SIZE = 50000
MIN_REPLAY = 1000
TARGET_UPDATE_FREQ = 1000  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000  # linear decay steps
TRAIN_FREQ = 4     # train every N environment steps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utilities / NN / Buffer
# -----------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[128, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------
# Agent class
# -----------------------
class TrafficLightAgent:
    def __init__(self, tl_id, lane_ids, state_dim, action_dim=2, min_green=DEFAULT_MIN_GREEN, max_green=DEFAULT_MAX_GREEN, yellow=DEFAULT_YELLOW):
        self.tl_id = tl_id
        self.lane_ids = lane_ids
        self.min_green = min_green
        self.max_green = max_green
        self.yellow = yellow
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self.replay = ReplayBuffer()
        self.steps_done = 0

    def select_action(self, state, eps):
        # state: numpy array
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax().item())

    def store_transition(self, *args):
        self.replay.push(*args)

    def can_train(self):
        return len(self.replay) >= MIN_REPLAY

    def optimize(self):
        if len(self.replay) < BATCH_SIZE:
            return 0.0
        batch = self.replay.sample(BATCH_SIZE)
        state = torch.tensor(np.array(batch.state), dtype=torch.float32, device=DEVICE)
        action = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=DEVICE)
        done = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)
        next_q = self.target_net(next_state).max(1)[0].detach().unsqueeze(1)
        expected_q = reward + (1.0 - done) * GAMMA * next_q

        loss = nn.functional.mse_loss(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# -----------------------
# Environment helpers
# -----------------------
def get_lane_metrics(lane_id):
    """Return (vehicle_count, waiting_time) for given lane id (safe calls)."""
    try:
        vcount = traci.lane.getLastStepVehicleNumber(lane_id)
        wtime = traci.lane.getWaitingTime(lane_id)
    except traci.TraCIException:
        # lane might not exist or be empty in some setups; return zeros
        vcount = 0
        wtime = 0.0
    return vcount, wtime

def get_tl_phase_info(tl_id):
    """Return (phase_index, time_in_phase, phase_duration)."""
    p_idx = traci.trafficlight.getPhase(tl_id)
    # time in phase: we can use getPhaseDuration and getNextSwitch? TraCI doesn't provide time_in_phase directly.
    # We'll track time_in_phase manually using a dict in the env loop.
    duration = traci.trafficlight.getPhaseDuration(tl_id)
    return p_idx, duration

# -----------------------
# State builder
# -----------------------
def build_state_for_tl(tl_cfg, time_in_phase_dict):
    """
    Build normalized state vector for a traffic light config.
    state vector layout:
      - vehicle counts for each lane (len = n_lanes)
      - waiting times for each lane (len = n_lanes)
      - one-hot phase vector (4 phases)
      - time_in_phase normalized by max_green
    """
    lanes = tl_cfg["lanes"]
    counts = []
    waits = []
    for lane in lanes:
        v, w = get_lane_metrics(lane)
        counts.append(v)
        waits.append(w)

    # basic normalizations (avoid huge values)
    max_count = 20.0
    max_wait = 200.0
    counts = [min(c / max_count, 1.0) for c in counts]
    waits = [min(w / max_wait, 1.0) for w in waits]

    tl_id = tl_cfg["id"]
    phase_idx = traci.trafficlight.getPhase(tl_id)
    # one-hot
    onehot = [0.0] * 4
    if 0 <= phase_idx < 4:
        onehot[phase_idx] = 1.0

    time_in_phase = time_in_phase_dict.get(tl_id, 0)
    time_norm = min(time_in_phase / DEFAULT_MAX_GREEN, 1.0)

    state = counts + waits + onehot + [time_norm]
    return np.array(state, dtype=np.float32)

# -----------------------
# Main training loop
# -----------------------
def run(mode="train", checkpoint=None, episodes=100):
    # prepare agents
    tl_ids = list(TL_CONFIGS.keys())
    tl_cfgs = {}
    for k, v in TL_CONFIGS.items():
        cfg = dict(v)  # copy
        cfg["id"] = k
        cfg["min_green"] = v.get("min_green", DEFAULT_MIN_GREEN)
        cfg["max_green"] = v.get("max_green", DEFAULT_MAX_GREEN)
        cfg["yellow"] = v.get("yellow", DEFAULT_YELLOW)
        tl_cfgs[k] = cfg

    # compute state dims (all agents share same structure but might have different lane counts)
    agents = {}
    for k, cfg in tl_cfgs.items():
        # counts + waits = 2 * n_lanes
        n_lanes = len(cfg["lanes"])
        state_dim = 2 * n_lanes + 4 + 1  # one-hot 4 phases + time_norm
        agent = TrafficLightAgent(k, cfg["lanes"], state_dim)
        agents[k] = agent

    # option to load checkpoint
    if checkpoint:
        data = torch.load(checkpoint, map_location=DEVICE)
        for k in agents:
            agents[k].policy_net.load_state_dict(data[k]["policy"])
            agents[k].target_net.load_state_dict(data[k]["target"])
            agents[k].optimizer.load_state_dict(data[k]["opt"])

    # Start SUMO via TraCI
    sumo_cmd = [SUMO_BINARY, "-c", SUMOCFG, "--step-length", str(STEP_LENGTH)]
    print("Starting SUMO with command:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)

    global_step = 0
    random.seed(0)
    np.random.seed(0)

    # track time_in_phase per tl manually (increment each step unless phase changed)
    time_in_phase = {k: 0 for k in tl_ids}
    last_phase = {k: traci.trafficlight.getPhase(k) for k in tl_ids}

    for ep in range(episodes):
        ep_rewards = {k: 0.0 for k in tl_ids}
        ep_waits = {k: 0.0 for k in tl_ids}

        # reset SUMO for new episode: to keep simple, we restart SUMO for each episode
        # (requires stopping and starting TraCI). To simplify runtime, here we use one long run
        # If you want per-episode resets, wrap traci.start/close per episode.
        if ep > 0:
            # restart simulation
            traci.load(["-c", SUMOCFG])

            # reset phase trackers
            time_in_phase = {k: 0 for k in tl_ids}
            last_phase = {k: traci.trafficlight.getPhase(k) for k in tl_ids}

        print(f"Episode {ep+1}/{episodes} -- starting")

        for step in range(EPISODE_STEPS):
            # for each tl build state, choose action, apply
            actions = {}
            states = {}
            rewards = {}
            next_states = {}
            dones = {}

            # update time_in_phase tracking
            for k in tl_ids:
                curr_phase = traci.trafficlight.getPhase(k)
                if curr_phase == last_phase[k]:
                    time_in_phase[k] += 1
                else:
                    time_in_phase[k] = 0
                    last_phase[k] = curr_phase

            # choose and apply actions for each agent
            eps = max(EPS_END, EPS_START - global_step * (EPS_START - EPS_END) / EPS_DECAY)

            for k, agent in agents.items():
                cfg = tl_cfgs[k]
                s = build_state_for_tl(cfg, time_in_phase)
                states[k] = s

                # policy
                a = agent.select_action(s, eps)
                actions[k] = a

                # apply action:
                phase_idx = traci.trafficlight.getPhase(k)
                t_in = time_in_phase[k]
                # determine if current phase is green (we assume phases 0 and 2 are green as in your tlLogic)
                is_green = (phase_idx % 2 == 0)  # 0 and 2 = green, 1 and 3 = yellow

                if a == 0:
                    # maintain: if green and not exceeding max, extend by 1 (increase remaining duration)
                    if is_green and t_in < cfg["max_green"]:
                        # set remaining duration = current duration + 1
                        # We compute new duration = current configured - elapsed + 1 added to remaining.
                        # Use setPhaseDuration to increase duration by 1.
                        try:
                            cur_dur = traci.trafficlight.getPhaseDuration(k)
                            # setPhaseDuration sets the duration of current phase (total length),
                            # so we set it to cur_dur + 1 to extend future duration by 1s (practical and simple).
                            new_dur = cur_dur + 1
                            traci.trafficlight.setPhaseDuration(k, new_dur)
                        except Exception:
                            pass  # if not supported, ignore
                    # else do nothing
                elif a == 1:
                    # change: only if min_green satisfied (or if not green, allow)
                    if is_green and t_in >= cfg["min_green"]:
                        # move to next phase
                        next_phase = (phase_idx + 1) % 4
                        try:
                            traci.trafficlight.setPhase(k, next_phase)
                            # optionally set duration of this (new) phase to its configured value (yellow, etc.)
                            # we won't override here to avoid complex mapping; SUMO's tlLogic durations remain.
                        except Exception:
                            pass
                    elif not is_green:
                        # if already in yellow/red, allow advance (let SUMO handle typical progression)
                        try:
                            next_phase = (phase_idx + 1) % 4
                            traci.trafficlight.setPhase(k, next_phase)
                        except Exception:
                            pass

            # advance simulation one step
            traci.simulationStep()

            # compute rewards and store transitions
            for k, agent in agents.items():
                cfg = tl_cfgs[k]
                # compute reward as negative sum of waiting time across lanes
                total_wait = 0.0
                for lane in cfg["lanes"]:
                    _, lane_wait = get_lane_metrics(lane)
                    total_wait += lane_wait
                reward = -total_wait
                ep_rewards[k] += reward
                ep_waits[k] += total_wait

                # next state
                ns = build_state_for_tl(cfg, time_in_phase)
                done = (step == EPISODE_STEPS - 1)

                agent.store_transition(states[k], actions[k], reward, ns, float(done))

                # periodic training
                if global_step % TRAIN_FREQ == 0 and agent.can_train() and mode == "train":
                    loss = agent.optimize()

                # update target net periodically
                if global_step % TARGET_UPDATE_FREQ == 0:
                    agent.update_target()

            global_step += 1

        # end of episode logs
        print(f"Episode {ep+1} finished.")
        for k in agents:
            avg_reward = ep_rewards[k] / EPISODE_STEPS
            avg_wait = ep_waits[k] / EPISODE_STEPS
            print(f"  {k}: avg reward step {avg_reward:.2f}, avg total wait {avg_wait:.2f}")

    # Save checkpoint
    ckpt = {}
    for k, a in agents.items():
        ckpt[k] = {
            "policy": a.policy_net.state_dict(),
            "target": a.target_net.state_dict(),
            "opt": a.optimizer.state_dict()
        }
    torch.save(ckpt, "checkpoint_multi_agent.pth")
    print("Saved checkpoint_multi_agent.pth")

    traci.close()
    print("SUMO closed. Done.")

# -----------------------
# CLI and run
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run(mode=args.mode, checkpoint=args.checkpoint, episodes=args.episodes)
