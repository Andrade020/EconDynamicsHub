"""
dqn_cournot_cpu.py

DQN multiagent para oligopólio Cournot (ações discretas).
Feito para rodar eficientemente em CPU (i7 12 cores, 32GB).

Requisitos:
    pip install numpy matplotlib torch

Rode:
    python dqn_cournot_cpu.py
"""

import math
import time
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Configuração do experimento
# -----------------------
SEED = int(time.time() * 1000) % (2**32 - 1)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cpu')  # explicitamente CPU

# Parâmetros econômicos (ambiente)
a = 100.0
b = 1.0
c = 20.0
N_AGENTS = 4           # número de firmas/agentes no mercado
q_max = (a - c) / b

# Discretização de ações (quantidades)
K = 31                 # número de ações discretas (ajuste: 21..61)
ACTIONS = np.linspace(0.0, q_max, K)

# Observação / estado
# Aqui usamos estado = vetor com [preço anterior, própria ação anterior, média rival anterior]
# Todos normalizados entre 0 e 1 para estabilidade do RL.
STATE_DIM = 3

# Parâmetros DQN / treinamento (ajuste para CPU)
N_ENVS = 8             # ambientes/episódios paralelos para coleta (ajuste para teu CPU)
EPISODES = 12000        # número de batches de coleta (cada batch contém T steps por env)
T = 16                 # passos por episódio (jogo repetido por episódio)
GAMMA = 0.98
LR = 5e-4
BATCH_SIZE = 128
REPLAY_SIZE = 200_000
MIN_REPLAY = 2000
TARGET_UPDATE_FREQ = 1000   # passos (de atualização do target network)
TRAIN_EVERY = 4             # treinar a cada X passos de coleta
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1e-5            # decaimento por passo (ta mais gradual)

# Rede pequena (eficiente em CPU)
HIDDEN = 64

# Salvar modelos/plots
SAVE_MODEL = True
MODEL_PATH = "dqn_agent_{}.pth"

# -----------------------
# Utilitários e ambiente
# -----------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def compute_price_and_profits(qs):
    """qs: array shape (N_AGENTS,)"""
    Q = float(np.sum(qs))
    P = max(a - b * Q, 0.0)
    profits = (P - c) * qs
    return P, profits

def normalize_state(p, own_q_prev, avg_others_prev):
    """Retorna vetor state normalizado entre 0 e 1"""
    # price in [0,a], own_q_prev in [0,q_max], avg_others_prev same
    return np.array([p / a, own_q_prev / q_max if q_max > 0 else 0.0, avg_others_prev / q_max if q_max > 0 else 0.0], dtype=np.float32)

class VectorizedCournotEnv:
    """
    Ambientes vetorizados: contém N_ENVS episódios paralelos.
    Estado por agente: price_prev, own_q_prev, avg_others_prev (para cada agente).
    Ao reset, inicializa preços e ações anteriores.
    step(actions_batch) -> next_states, rewards, dones, infos
      actions_batch shape: (n_envs, N_AGENTS) indices de ação (0..K-1)
    """
    def __init__(self, n_envs, n_agents, actions_array):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.actions_array = actions_array
        self.K = len(actions_array)
        self.reset()

    def reset(self):
        # initialize previous price around c, and q prev small random
        price_prev = np.full(self.n_envs, c)  # start at cost
        own_prev = np.zeros((self.n_envs, self.n_agents))
        # compute avg others
        avg_others_prev = np.zeros((self.n_envs, self.n_agents))
        states = np.zeros((self.n_envs, self.n_agents, STATE_DIM), dtype=np.float32)
        for e in range(self.n_envs):
            for i in range(self.n_agents):
                states[e, i] = normalize_state(price_prev[e], own_prev[e, i], avg_others_prev[e, i])
        self.price_prev = price_prev
        self.own_prev = own_prev
        self.avg_others_prev = avg_others_prev
        # dones per env (we use episodes of fixed length externally)
        return states

    def step(self, actions_idx):
        """
        actions_idx: shape (n_envs, n_agents) actions indices
        returns:
          next_states: (n_envs, n_agents, STATE_DIM)
          rewards: (n_envs, n_agents)
          dones: (n_envs,) - we will manage episode length outside
        """
        n = self.n_envs
        rewards = np.zeros((n, self.n_agents), dtype=np.float32)
        next_states = np.zeros((n, self.n_agents, STATE_DIM), dtype=np.float32)
        prices = np.zeros(n, dtype=np.float32)

        for e in range(n):
            # get chosen quantities
            qs = self.actions_array[actions_idx[e]]  # shape (n_agents,)
            P, profits = compute_price_and_profits(qs)
            prices[e] = P
            # update rewards
            rewards[e, :] = profits.astype(np.float32)
            # update prev arrays for next state
            avg_others = (qs.sum() - qs) / max(1, (self.n_agents - 1))
            for i in range(self.n_agents):
                next_states[e, i] = normalize_state(P, qs[i], avg_others[i])

            # store for internal (not strictly necessary if external manages state)
            self.price_prev[e] = P
            self.own_prev[e, :] = qs
            self.avg_others_prev[e, :] = avg_others

        dones = np.zeros(n, dtype=np.bool_)  # episodes end externally
        return next_states, rewards, dones, {'prices': prices}

# -----------------------
# Replay buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------
# DQN network
# -----------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=HIDDEN):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Agent (independent DQN)
# -----------------------
class DQNAgent:
    def __init__(self, id, state_dim, n_actions, lr=LR):
        self.id = id
        self.n_actions = n_actions
        self.q_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(REPLAY_SIZE)
        self.learn_steps = 0

    def act(self, states_np, epsilon):
        # states_np shape: (n_envs, state_dim)
        n = states_np.shape[0]
        if random.random() < epsilon:
            # random actions per env
            return np.random.randint(0, self.n_actions, size=n)
        # else greedy (batch)
        with torch.no_grad():
            s = torch.from_numpy(states_np).float().to(device)  # (n, state_dim)
            qvals = self.q_net(s)  # (n, n_actions)
            actions = qvals.argmax(dim=1).cpu().numpy()
            return actions

    def store_transitions(self, transitions):
        # transitions: list of Transition tuples from vectorized envs
        for tr in transitions:
            self.replay.push(tr.state, tr.action, tr.reward, tr.next_state, tr.done)

    def sample_and_learn(self, batch_size=BATCH_SIZE):
        if len(self.replay) < max(MIN_REPLAY, batch_size):
            return None
        trans = self.replay.sample(batch_size)
        states = torch.tensor(np.array(trans.state), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(trans.action), dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(np.array(trans.reward), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(trans.next_state), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(trans.done), dtype=torch.float32).unsqueeze(1).to(device)

        q_values = self.q_net(states).gather(1, actions)  # (batch,1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            td_target = rewards + (1.0 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping small to stabilize
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

# -----------------------
# Treinamento multi-agente (loop principal)
# -----------------------
def train():
    start_time = time.time()
    n_envs = N_ENVS
    env = VectorizedCournotEnv(n_envs, N_AGENTS, ACTIONS)
    # Create one DQNAgent per firm
    agents = [DQNAgent(i, STATE_DIM, K) for i in range(N_AGENTS)]

    # epsilon schedule (by steps)
    epsilon = EPS_START
    global_steps = 0

    # métricas
    mean_q_per_batch = []
    mean_p_per_batch = []
    losses = []

    # inicializa estados
    states = env.reset()  # shape (n_envs, N_AGENTS, STATE_DIM)

    for batch in range(EPISODES):
        # cada batch corresponde a coletar T steps em N_ENVS episódios paralelos
        batch_qs = []
        batch_ps = []
        for t in range(T):
            # construir ações para todos agentes/envs
            actions_idx = np.zeros((n_envs, N_AGENTS), dtype=np.int64)
            for ag in range(N_AGENTS):
                # states[:, ag, :] shape -> (n_envs, state_dim)
                states_ag = states[:, ag, :]
                actions_ag = agents[ag].act(states_ag, epsilon)
                actions_idx[:, ag] = actions_ag

            # step env
            next_states, rewards, dones, info = env.step(actions_idx)
            prices = info['prices']  # shape (n_envs,)

            # armazenar transições por agente (um Transition por env por agente)
            for ag in range(N_AGENTS):
                # compile transitions of this agent across envs
                s = states[:, ag, :].astype(np.float32)
                a = actions_idx[:, ag].astype(np.int32)
                r = rewards[:, ag].astype(np.float32)
                ns = next_states[:, ag, :].astype(np.float32)
                d = dones.astype(np.float32)
                # push per env
                for e in range(n_envs):
                    tr = Transition(s[e], int(a[e]), float(r[e]), ns[e], float(d[e]))
                    agents[ag].replay.push(*tr)

            # advance
            states = next_states

            # métricas por passo
            # média de q no passo (sobre envs e agentes)
            qs_step = ACTIONS[actions_idx].mean()
            batch_qs.append(qs_step)
            batch_ps.append(prices.mean())

            global_steps += 1
            # epsilon decay
            epsilon = max(EPS_END, epsilon - EPS_DECAY)

            # treinar agentes periodicamente
            if global_steps % TRAIN_EVERY == 0:
                for ag in range(N_AGENTS):
                    loss = agents[ag].sample_and_learn()
                    if loss is not None:
                        losses.append(loss)

        # métricas por batch
        mean_q_per_batch.append(np.mean(batch_qs))
        mean_p_per_batch.append(np.mean(batch_ps))

        if (batch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"[Batch {batch+1}/{EPISODES}] time {elapsed:.1f}s - eps {epsilon:.3f} - mean_q {mean_q_per_batch[-1]:.3f} - mean_p {mean_p_per_batch[-1]:.3f} - replay sizes {[len(a.replay) for a in agents]}")

    # salvar modelos
    if SAVE_MODEL:
        for ag in range(N_AGENTS):
            torch.save(agents[ag].q_net.state_dict(), MODEL_PATH.format(ag))

    return mean_q_per_batch, mean_p_per_batch, losses, agents

# -----------------------
# Execução
# -----------------------
if __name__ == "__main__":
    t0 = time.time()
    mq, mp, losses, agents = train()
    print("Treinamento finalizado em {:.1f}s".format(time.time() - t0))

    # Benchmarks teóricos (por firma)
    q_cournot = (a - c) / (b * (N_AGENTS + 1))
    q_collusion = (a - c) / (2 * b * N_AGENTS)
    q_perfect = (a - c) / (b * N_AGENTS)
    P_cournot = a - b * N_AGENTS * q_cournot
    P_collusion = a - b * N_AGENTS * q_collusion
    P_perfect = c

    # Plots
    plt.figure(figsize=(10,5))
    plt.plot(mq, label='Quantidade média por batch')
    plt.axhline(q_cournot, color='red', linestyle='--', label='Cournot')
    plt.axhline(q_collusion, color='green', linestyle=':', label='Colusão')
    plt.axhline(q_perfect, color='blue', linestyle='-.', label='Concorrência perfeita')
    plt.xlabel('Batch')
    plt.ylabel('Quantidade média por firma')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(mp, label='Preço médio por batch')
    plt.axhline(P_cournot, color='red', linestyle='--', label='P Cournot')
    plt.axhline(P_collusion, color='green', linestyle=':', label='P Colusão')
    plt.axhline(P_perfect, color='blue', linestyle='-.', label='P Concorrência perfeita')
    plt.xlabel('Batch')
    plt.ylabel('Preço médio')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print política final média por agente (ação preferida média)
    for ag in range(N_AGENTS):
        q_pref_idx = agents[ag].q_net(torch.from_numpy(np.zeros((1, STATE_DIM), dtype=np.float32))).argmax().item()
        print(f"Agente {ag}: ação preferida (index) {q_pref_idx} q={ACTIONS[q_pref_idx]:.3f}")
