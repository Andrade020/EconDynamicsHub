# hybrid_evo_rl_cournot.py
"""
Híbrido: Evolução populacional + Aprendizado por Reforço (DQN curto durante a vida)
CPU-friendly, vetorizado, com avaliação paralela.

Requisitos:
    pip install numpy torch matplotlib joblib

Rode:
    python hybrid_evo_rl_cournot.py
"""

import time
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# -----------------------
# Seeds / dispositivo
# -----------------------
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cpu')  # explicitamente CPU

# -----------------------
# Parâmetros econômicos
# -----------------------
a = 100.0
b = 1.0
c = 20.0
N_PLAYERS_PER_MARKET = 4
q_max = (a - c) / b

# -----------------------
# Ações discretas (quantidades)
# -----------------------
K = 31
ACTIONS = np.linspace(0.0, q_max, K)

# -----------------------
# Benchmarks analíticos
# -----------------------
q_cournot = (a - c) / (b * (N_PLAYERS_PER_MARKET + 1))
q_collusion = (a - c) / (2 * b * N_PLAYERS_PER_MARKET)
q_perfect = (a - c) / (b * N_PLAYERS_PER_MARKET)

P_cournot = a - b * N_PLAYERS_PER_MARKET * q_cournot
P_collusion = a - b * N_PLAYERS_PER_MARKET * q_collusion
P_perfect = c

# -----------------------
# Parâmetros de RL (por indivíduo)
# -----------------------
STATE_DIM = 3   # [price_prev, own_q_prev, avg_others_prev] (normalizados)
HIDDEN = 64
LR = 5e-4
GAMMA = 0.97
BATCH_SIZE = 64
REPLAY_CAP = 20000
MIN_REPLAY = 400
TARGET_UPDATE = 500
TRAIN_EVERY = 4   # cada quantas participações o agente tenta aprender

# -----------------------
# Parâmetros evolutivos / execução
# -----------------------
POP_SIZE = 40            # população de agentes (ajuste: 20..80)
GENERATIONS = 30         # número de gerações (ajuste)
LIFE_STEPS = 2000        # passos de vida por geração (cada passo = N_ENVS mercados)
N_ENVS = 12              # quantos mercados paralelos por passo (use ~n_cores)
EVAL_EPISODES = 12       # episódios para avaliação da elite
TOURNAMENT = 3
ELITISM = 2
MUTATION_STD = 0.02
NEW_RANDOM_PROB = 0.02

# -----------------------
# Utilitários
# -----------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def compute_price_and_profits(qs):
    """qs: np.array shape (n_players,)"""
    Q = float(np.sum(qs))
    P = max(a - b * Q, 0.0)
    profits = (P - c) * qs
    return P, profits

def normalize_state(p, own_q_prev, avg_others_prev):
    """state normalized to ~[0,1]"""
    return np.array([p / a, own_q_prev / q_max if q_max>0 else 0.0, avg_others_prev / q_max if q_max>0 else 0.0], dtype=np.float32)

# -----------------------
# Ambiente vetorizado
# -----------------------
class VectorizedCournotEnv:
    def __init__(self, n_envs, n_players, actions_array):
        self.n_envs = n_envs
        self.n_players = n_players
        self.actions_array = actions_array
        self.K = len(actions_array)
        self.reset()

    def reset(self):
        # start price at c, previous q = 0
        self.price_prev = np.full(self.n_envs, c, dtype=np.float32)
        self.own_prev = np.zeros((self.n_envs, self.n_players), dtype=np.float32)
        self.avg_others_prev = np.zeros((self.n_envs, self.n_players), dtype=np.float32)
        states = np.zeros((self.n_envs, self.n_players, STATE_DIM), dtype=np.float32)
        for e in range(self.n_envs):
            for i in range(self.n_players):
                states[e, i] = normalize_state(self.price_prev[e], 0.0, 0.0)
        self.t = 0
        return states

    def step(self, actions_idx):
        """
        actions_idx: (n_envs, n_players) indices
        returns next_states (n_envs, n_players, state_dim), rewards (n_envs, n_players), dones, info
        """
        n = self.n_envs
        rewards = np.zeros((n, self.n_players), dtype=np.float32)
        next_states = np.zeros((n, self.n_players, STATE_DIM), dtype=np.float32)
        prices = np.zeros(n, dtype=np.float32)

        for e in range(n):
            qs = self.actions_array[actions_idx[e]]   # shape (n_players,)
            P, profits = compute_price_and_profits(qs)
            prices[e] = P
            rewards[e, :] = profits.astype(np.float32)
            avg_others = (qs.sum() - qs) / max(1, (self.n_players - 1))
            for i in range(self.n_players):
                next_states[e, i] = normalize_state(P, qs[i], avg_others[i])
            self.price_prev[e] = P
            self.own_prev[e, :] = qs
            self.avg_others_prev[e, :] = avg_others

        self.t += 1
        dones = np.zeros(n, dtype=np.bool_)
        return next_states, rewards, dones, {'prices': prices, 'q': self.actions_array[actions_idx]}

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
# Rede Q (DQN)
# -----------------------
class QNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=HIDDEN):
        super().__init__()
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
# Agente evolutivo com DQN (aprende durante a vida)
# -----------------------
class EvoAgent:
    def __init__(self, idx, state_dim=STATE_DIM, n_actions=K, lr=LR):
        self.id = idx
        self.n_actions = n_actions
        self.q_net = QNet(state_dim, n_actions).to(device)
        self.target_net = QNet(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(REPLAY_CAP)
        self.learn_steps = 0
        self.participation = 0

    def act(self, state, epsilon=0.1):
        """state: numpy array (state_dim,)"""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            qvals = self.q_net(s)
            return int(qvals.argmax(dim=1).item())

    def store(self, s, a, r, ns, done):
        self.replay.push(s, int(a), float(r), ns, float(done))
        self.participation += 1

    def sample_and_learn(self, batch_size=BATCH_SIZE):
        if len(self.replay) < max(MIN_REPLAY, batch_size):
            return None
        trans = self.replay.sample(batch_size)
        states = torch.tensor(np.array(trans.state), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(trans.action), dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(np.array(trans.reward), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(trans.next_state), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(trans.done), dtype=torch.float32).unsqueeze(1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            td_target = rewards + (1.0 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(q_values, td_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def clone(self, new_id):
        child = EvoAgent(new_id)
        child.q_net.load_state_dict(copy.deepcopy(self.q_net.state_dict()))
        child.target_net.load_state_dict(copy.deepcopy(self.target_net.state_dict()))
        return child

    def mutate(self, std=MUTATION_STD):
        with torch.no_grad():
            for p in self.q_net.parameters():
                if random.random() < 0.5:
                    noise = torch.randn_like(p) * std
                    p.add_(noise)
        self.target_net.load_state_dict(self.q_net.state_dict())

# -----------------------
# Avaliação: retorna (avg_profit_per_player, avg_q, avg_price)
# Eval pairs agent with clones of itself (measures payoff when policy adopted by all)
# -----------------------
def evaluate_agent_full(agent, n_envs=12, steps=12):
    env = VectorizedCournotEnv(n_envs, N_PLAYERS_PER_MARKET, ACTIONS)
    states = env.reset()
    total_profits = 0.0
    total_q = 0.0
    total_price = 0.0
    count = 0
    for _ in range(steps):
        actions_idx = np.zeros((n_envs, N_PLAYERS_PER_MARKET), dtype=np.int64)
        for e in range(n_envs):
            for p in range(N_PLAYERS_PER_MARKET):
                s = states[e, p]
                actions_idx[e, p] = agent.act(s, epsilon=0.0)
        ns, rewards, dones, info = env.step(actions_idx)
        total_profits += rewards.sum()
        total_q += info['q'].mean()
        total_price += info['prices'].mean()
        count += (n_envs * N_PLAYERS_PER_MARKET)
        states = ns
    avg_profit = total_profits / count
    avg_q = total_q / steps
    avg_price = total_price / steps
    return avg_profit, float(avg_q), float(avg_price)

# -----------------------
# Loop principal: vida (aprendizado) + avaliação + seleção
# -----------------------
def run_hybrid_evo():
    # init population
    population = [EvoAgent(i) for i in range(POP_SIZE)]

    mean_q_history = []
    mean_p_history = []
    fitness_history = []
    start = time.time()

    for gen in range(GENERATIONS):
        t0 = time.time()
        print(f"\n--- Geração {gen+1}/{GENERATIONS} ---")
        # LIFE: cada geração tem LIFE_STEPS passos; em cada passo criamos N_ENVS mercados
        env = VectorizedCournotEnv(N_ENVS, N_PLAYERS_PER_MARKET, ACTIONS)
        states = env.reset()
        # participants per env sampled each step from population
        for step in range(LIFE_STEPS):
            # sample participants for each env (without replacement in a market)
            participants = np.zeros((N_ENVS, N_PLAYERS_PER_MARKET), dtype=int)
            for e in range(N_ENVS):
                participants[e] = np.random.choice(len(population), size=N_PLAYERS_PER_MARKET, replace=False)

            # get actions
            actions_idx = np.zeros((N_ENVS, N_PLAYERS_PER_MARKET), dtype=np.int64)
            for e in range(N_ENVS):
                for p in range(N_PLAYERS_PER_MARKET):
                    agent_idx = participants[e, p]
                    agent = population[agent_idx]
                    s = states[e, p]
                    # epsilon schedule during life: start higher then decay
                    eps = max(0.02, 0.6 * (1.0 - step / LIFE_STEPS))
                    actions_idx[e, p] = agent.act(s, epsilon=eps)

            # step
            next_states, rewards, dones, info = env.step(actions_idx)

            # store transitions per participating agent and train occasionally
            for e in range(N_ENVS):
                for p in range(N_PLAYERS_PER_MARKET):
                    agent_idx = participants[e, p]
                    agent = population[agent_idx]
                    s = states[e, p]
                    a = int(actions_idx[e, p])
                    r = float(rewards[e, p])
                    ns = next_states[e, p]
                    d = False
                    agent.store(s, a, r, ns, d)
                    # learning (per-agent schedule)
                    if agent.participation % TRAIN_EVERY == 0:
                        _ = agent.sample_and_learn()

            states = next_states

            # lightweight logging
            if (step+1) % (LIFE_STEPS // 4) == 0:
                print(f" life step {step+1}/{LIFE_STEPS} elapsed {time.time()-t0:.1f}s")

        # After life: evaluate population (parallel)
        print(" Avaliando população (paralelo)...")
        results = Parallel(n_jobs=min(12, POP_SIZE))(delayed(evaluate_agent_full)(population[i], n_envs=12, steps=12) for i in range(len(population)))
        # results: list of tuples (avg_profit, avg_q, avg_price)
        profits = np.array([r[0] for r in results], dtype=float)
        qs = np.array([r[1] for r in results], dtype=float)
        ps = np.array([r[2] for r in results], dtype=float)

        # record stats: mean/elite
        best_idx = int(np.argmax(profits))
        mean_fit = float(np.mean(profits))
        best_fit = float(profits[best_idx])
        fitness_history.append(mean_fit)
        mean_q_history.append(float(qs[best_idx]))   # quantity of best policy (mean per market)
        mean_p_history.append(float(ps[best_idx]))
        print(f" gen {gen+1} best profit {best_fit:.4f} mean profit {mean_fit:.4f} best idx {best_idx} q_best {qs[best_idx]:.3f} p_best {ps[best_idx]:.3f}")

        # SELECTION: elitism + tournament reproduction + mutation + random new
        sorted_idx = np.argsort(-profits)
        new_pop = []
        # elitism: copy top ELITISM
        for k in range(ELITISM):
            idx = sorted_idx[k]
            child = population[idx].clone(new_id=k)
            new_pop.append(child)

        # fill the rest
        next_id = ELITISM
        while len(new_pop) < POP_SIZE:
            # tournament among population indices
            cand = np.random.choice(range(len(population)), size=TOURNAMENT, replace=False)
            cand_best = cand[int(np.argmax(profits[cand]))]
            if random.random() < NEW_RANDOM_PROB:
                # fresh random agent
                ind = EvoAgent(next_id)
            else:
                ind = population[cand_best].clone(new_id=next_id)
                ind.mutate(std=MUTATION_STD)
            new_pop.append(ind)
            next_id += 1

        population = new_pop
        print(f" Geração {gen+1} completa em {time.time()-t0:.1f}s; total elapsed {time.time()-start:.1f}s")

    return population, fitness_history, mean_q_history, mean_p_history

# -----------------------
# Execução principal
# -----------------------
if __name__ == "__main__":
    t_start = time.time()
    pop, fitness_hist, q_hist, p_hist = run_hybrid_evo()
    print(f"\nExecução completa em {time.time()-t_start:.1f}s")

    # Plots
    gens = list(range(1, len(q_hist)+1))

    plt.figure(figsize=(10,4))
    plt.plot(gens, q_hist, marker='o', label='Q médio (elite)')
    plt.axhline(q_cournot, color='red', linestyle='--', label='Q Cournot')
    plt.axhline(q_collusion, color='green', linestyle=':', label='Q Colusão')
    plt.axhline(q_perfect, color='blue', linestyle='-.', label='Q Concorrência perfeita')
    plt.xlabel('Geração')
    plt.ylabel('Quantidade média por firma')
    plt.title('Evolução da quantidade média (elite)')
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(gens, p_hist, marker='o', label='P médio (elite)')
    plt.axhline(P_cournot, color='red', linestyle='--', label='P Cournot')
    plt.axhline(P_collusion, color='green', linestyle=':', label='P Colusão')
    plt.axhline(P_perfect, color='blue', linestyle='-.', label='P Concorrência perfeita')
    plt.xlabel('Geração')
    plt.ylabel('Preço médio')
    plt.title('Evolução do preço médio (elite)')
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(gens, fitness_hist, marker='o', label='Fitness médio (population)')
    plt.xlabel('Geração'); plt.ylabel('Lucro médio por firma')
    plt.title('Evolução do fitness médio')
    plt.grid(True); plt.legend()
    plt.show()
