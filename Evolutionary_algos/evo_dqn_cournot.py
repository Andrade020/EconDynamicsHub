"""
evo_dqn_cournot.py

Combinação: Evolução populacional + DQN (Lamarckian) para oligopólio Cournot.
Projetado para CPU (i7 12 cores, 32GB).

Requisitos:
    pip install numpy torch matplotlib

Rode:
    python evo_dqn_cournot.py
"""

import time
import random
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# -------------------------
# Configurações principais
# -------------------------
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cpu')  # explicitamente CPU

# --- Ambiente econômico ---
a = 100.0
b = 1.0
c = 20.0
N_PLAYERS_PER_MARKET = 4   # número de firmas por mercado (N)
q_max = (a - c) / b

# --- Ações discretas ---
K = 31  # número de ações discretas (quantidades)
ACTIONS = np.linspace(0.0, q_max, K)

# --- Observação / estado ---
# estado simples: [price_prev, own_q_prev, avg_others_prev] normalizados
STATE_DIM = 3

# --- Parâmetros DQN (por indivíduo) ---
HIDDEN = 64
LR = 5e-4
GAMMA = 0.98
BATCH_SIZE = 64
REPLAY_SIZE = 20000
MIN_REPLAY = 500
TARGET_UPDATE_FREQ = 500
TRAIN_EVERY = 4  # treina a cada X passos em que o indivíduo participou

# --- Parâmetros evolutivos ---
POP_SIZE = 40             # tamanho da população
GENERATIONS = 20          # número de gerações
LIFE_STEPS = 2000         # passos por geração (cada passo = um conjunto de N_ENVS mercados)
N_ENVS = 8                # mercados paralelos por passo
EVAL_EPISODES = 16        # episódios p/ avaliação final por indivíduo
TOURNAMENT_SIZE = 3
ELITISM = 2               # número de indivíduos que sobrevivem sem alteração
MUTATION_STD = 0.02       # desvio padrão da mutação de parâmetros (multiplicador)
NEW_RANDOM_PROB = 0.02    # probabilidade de criar novo aleatório ao invés de reproduzir

# -------------------------
# Utilitários e ambiente
# -------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def compute_price_and_profits(qs):
    """qs: array shape (n_players,)"""
    Q = float(np.sum(qs))
    P = max(a - b * Q, 0.0)
    profits = (P - c) * qs
    return P, profits

def normalize_state(p, own_q_prev, avg_others_prev):
    """Retorna vetor state normalizado (0..1)"""
    return np.array([p / a, own_q_prev / q_max if q_max > 0 else 0.0, avg_others_prev / q_max if q_max > 0 else 0.0], dtype=np.float32)

class VectorizedCournotEnv:
    """
    Vetorizado para coletar N_ENVS mercados em paralelo por passo.
    Cada mercado tem N_PLAYERS_PER_MARKET firmas (agentes participantes são passados).
    State por agente: [price_prev, own_q_prev, avg_others_prev].
    """
    def __init__(self, n_envs, n_players_per_market, actions_array):
        self.n_envs = n_envs
        self.n_players = n_players_per_market
        self.actions_array = actions_array
        self.K = len(actions_array)
        self.reset()

    def reset(self):
        self.price_prev = np.full(self.n_envs, c, dtype=np.float32)  # start at cost
        self.own_prev = np.zeros((self.n_envs, self.n_players), dtype=np.float32)
        self.avg_others_prev = np.zeros((self.n_envs, self.n_players), dtype=np.float32)
        states = np.zeros((self.n_envs, self.n_players, STATE_DIM), dtype=np.float32)
        for e in range(self.n_envs):
            for i in range(self.n_players):
                states[e, i] = normalize_state(self.price_prev[e], self.own_prev[e, i], self.avg_others_prev[e, i])
        return states

    def step(self, actions_idx):
        """
        actions_idx: shape (n_envs, n_players) -> indices das ações
        retorna: next_states (n_envs, n_players, state_dim), rewards (n_envs, n_players), infos
        """
        n = self.n_envs
        rewards = np.zeros((n, self.n_players), dtype=np.float32)
        next_states = np.zeros((n, self.n_players, STATE_DIM), dtype=np.float32)
        prices = np.zeros(n, dtype=np.float32)

        for e in range(n):
            qs = self.actions_array[actions_idx[e]]  # shape (n_players,)
            P, profits = compute_price_and_profits(qs)
            prices[e] = P
            rewards[e, :] = profits.astype(np.float32)
            avg_others = (qs.sum() - qs) / max(1, (self.n_players - 1))
            for i in range(self.n_players):
                next_states[e, i] = normalize_state(P, qs[i], avg_others[i])

            # store for potential debugging
            self.price_prev[e] = P
            self.own_prev[e, :] = qs
            self.avg_others_prev[e, :] = avg_others

        dones = np.zeros(n, dtype=np.bool_)
        return next_states, rewards, dones, {'prices': prices}

# -------------------------
# Replay buffer (por indivíduo)
# -------------------------
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

# -------------------------
# Rede DQN simples
# -------------------------
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

# -------------------------
# Indivíduo (agente DQN)
# -------------------------
class EvoDQNAgent:
    def __init__(self, id, state_dim=STATE_DIM, n_actions=K, lr=LR):
        self.id = id
        self.n_actions = n_actions
        self.q_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net = QNetwork(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(REPLAY_SIZE)
        self.learn_steps = 0
        self.participation_count = 0  # quantos passos participou na vida (para stats)

    def act(self, state_np, epsilon=0.0):
        """state_np: vetor shape (state_dim,) ou (batch, state_dim). Retorna índice(s)."""
        if state_np.ndim == 1:
            if random.random() < epsilon:
                return random.randrange(self.n_actions)
            s = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                qv = self.q_net(s)
                return int(qv.argmax(dim=1).cpu().item())
        else:
            # batch version
            batch = torch.from_numpy(state_np.astype(np.float32)).to(device)
            with torch.no_grad():
                qv = self.q_net(batch)  # (batch, n_actions)
                return qv.argmax(dim=1).cpu().numpy()

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, int(action), float(reward), next_state, float(done))
        self.participation_count += 1

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
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def get_param_vector(self):
        """Retorna parâmetros concatenados numpy (para mutação)"""
        vec = []
        for p in self.q_net.parameters():
            vec.append(p.detach().cpu().numpy().ravel())
        if len(vec) == 0:
            return np.array([])
        return np.concatenate(vec)

    def set_param_vector(self, vec):
        """Define parâmetros a partir de vetor 1D (mesma ordem de get_param_vector)."""
        ptr = 0
        for p in self.q_net.parameters():
            num = p.numel()
            arr = vec[ptr:ptr+num].reshape(p.size())
            p.data.copy_(torch.from_numpy(arr).to(device))
            ptr += num
        # atualizar target
        self.target_net.load_state_dict(self.q_net.state_dict())

    def mutate_parameters(self, std=MUTATION_STD):
        """Adiciona ruído gaussiano aos parâmetros (in-place)."""
        with torch.no_grad():
            for p in self.q_net.parameters():
                noise = torch.randn_like(p) * std
                p.add_(noise)
        # atualizar target
        self.target_net.load_state_dict(self.q_net.state_dict())

    def clone(self, new_id):
        """Retorna uma cópia profunda (pesos copiados)."""
        child = EvoDQNAgent(new_id)
        child.q_net.load_state_dict(copy.deepcopy(self.q_net.state_dict()))
        child.target_net.load_state_dict(copy.deepcopy(self.target_net.state_dict()))
        return child

# -------------------------
# Evolução + ciclo de vida
# -------------------------
def evaluate_agent(agent, n_episodes=EVAL_EPISODES, n_envs=8, steps_per_episode=8):
    """
    Avalia agent sozinho (ou em grupos randomizados?) — aqui avalia-se em combinação com N_PLAYERS_PER_MARKET-1
    other agents = copies que jogam com ação média; porém para avaliar fitness de indivíduo isolado num mercado
    precisamos montá-lo em mercados com rivais (que podem ser amostras aleatórias da população).
    Para simplicidade: durante avaliação, emparelhamos o agente com copys "baseline" (exceto preferimos
    avaliar com cópias do próprio agente para ver payoff quando todos usam essa política).
    """
    # Vamos avaliar o agent em mercados em que todos os participantes são cópias idênticas do agent.
    # Isso dá medida do payoff quando essa política é adotada por todos (cooperative payoff).
    total_profit = 0.0
    count = 0
    env = VectorizedCournotEnv(n_envs, N_PLAYERS_PER_MARKET, ACTIONS)
    for ep in range(n_episodes):
        states = env.reset()
        for t in range(steps_per_episode):
            # all players choose actions using greedy (epsilon=0)
            actions_idx = np.zeros((n_envs, N_PLAYERS_PER_MARKET), dtype=np.int64)
            for e in range(n_envs):
                for i in range(N_PLAYERS_PER_MARKET):
                    s = states[e, i]
                    ai = agent.act(s, epsilon=0.0)
                    actions_idx[e, i] = ai
            next_states, rewards, dones, info = env.step(actions_idx)
            # rewards shape (n_envs, n_players)
            total_profit += rewards.sum()
            count += rewards.size
            states = next_states
    avg_profit_per_player = total_profit / count if count > 0 else 0.0
    return avg_profit_per_player

def run_evolution():
    # Inicializa população
    population = [EvoDQNAgent(i) for i in range(POP_SIZE)]
    best_history = []
    mean_history = []
    price_history = []

    for gen in range(GENERATIONS):
        t0 = time.time()
        print(f"\n=== Geração {gen+1}/{GENERATIONS} ===")
        # --- VIDA: treinar durante LIFE_STEPS (cada passo = N_ENVS mercados paralelos) ---
        env = VectorizedCournotEnv(N_ENVS, N_PLAYERS_PER_MARKET, ACTIONS)
        states = env.reset()  # shape (n_envs, n_players, state_dim)

        # We'll randomly select participants for each market from population at each step.
        # participants_idx shape will be (n_envs, n_players)
        for step in range(LIFE_STEPS):
            # sample participants
            participants_idx = np.zeros((N_ENVS, N_PLAYERS_PER_MARKET), dtype=int)
            for e in range(N_ENVS):
                # sample without replacement from population indices
                participants_idx[e] = np.random.choice(range(len(population)), size=N_PLAYERS_PER_MARKET, replace=False)

            # actions selection for each market: build actions_idx (n_envs, n_players)
            actions_idx = np.zeros((N_ENVS, N_PLAYERS_PER_MARKET), dtype=np.int64)
            for e in range(N_ENVS):
                for p in range(N_PLAYERS_PER_MARKET):
                    agent_idx = participants_idx[e, p]
                    agent = population[agent_idx]
                    s = states[e, p]
                    # eps-greedy exploration schedule inside life: start higher, decay over life
                    # We'll use per-generation epsilon that decays slowly
                    eps = max(0.05, 0.6 * (1.0 - step / LIFE_STEPS))  # heuristic
                    ai = agent.act(s, epsilon=eps)
                    actions_idx[e, p] = ai

            # step env
            next_states, rewards, dones, info = env.step(actions_idx)

            # store transitions to each participating agent's replay and optionally train them
            for e in range(N_ENVS):
                for p in range(N_PLAYERS_PER_MARKET):
                    agent_idx = participants_idx[e, p]
                    agent = population[agent_idx]
                    s = states[e, p]
                    a = int(actions_idx[e, p])
                    r = float(rewards[e, p])
                    ns = next_states[e, p]
                    d = False
                    agent.store(s, a, r, ns, d)

                    # train occasionally (per-individual)
                    if agent.participation_count % TRAIN_EVERY == 0:
                        _ = agent.sample_and_learn()

            states = next_states

            # optionally log some step-level metrics every X steps
            if (step + 1) % (LIFE_STEPS // 5) == 0:
                elapsed = time.time() - t0
                print(f" gen {gen+1} life step {step+1}/{LIFE_STEPS} elapsed {elapsed:.1f}s")

        # --- AVALIAÇÃO: compute fitness for each individual ---
        fitness = np.zeros(len(population), dtype=np.float32)
        print(" Avaliando população (sem exploração)...")
        for i, agent in enumerate(population):
            # avaliação em múltiplos episódios: aqui usamos avaliação onde todos players em mercado usam a mesma política
            f = evaluate_agent(agent, n_episodes=EVAL_EPISODES, n_envs=8, steps_per_episode=8)
            fitness[i] = f
            if (i+1) % max(1, POP_SIZE//5) == 0:
                print(f"  avaliou {i+1}/{len(population)} - fitness {f:.4f}")

        # record stats
        best_idx = int(np.argmax(fitness))
        mean_fit = float(np.mean(fitness))
        print(f" Geração {gen+1} fitness: best {fitness[best_idx]:.4f} mean {mean_fit:.4f} best_id {best_idx}")
        best_history.append(float(fitness[best_idx]))
        mean_history.append(mean_fit)

        # --- SELEÇÃO + REPRODUÇÃO ---
        # Elitismo
        new_population = []
        sorted_idx = np.argsort(-fitness)  # descendente
        for k in range(ELITISM):
            elite_idx = sorted_idx[k]
            child = population[elite_idx].clone(new_id=k)
            new_population.append(child)

        # Fill rest by tournament selection + mutation (Lamarckian: copy weights)
        next_id = ELITISM
        while len(new_population) < POP_SIZE:
            # tournament selection among population indices
            candidates = np.random.choice(range(len(population)), size=TOURNAMENT_SIZE, replace=False)
            cand_fits = fitness[candidates]
            winner = candidates[int(np.argmax(cand_fits))]
            if random.random() < NEW_RANDOM_PROB:
                # create new random individual
                ind = EvoDQNAgent(next_id)
            else:
                ind = population[winner].clone(new_id=next_id)
                # mutate parameters by adding gaussian noise scaled by parameter magnitudes
                ind.mutate_parameters(std=MUTATION_STD)
            new_population.append(ind)
            next_id += 1

        # swap population
        population = new_population

        # record price proxy: we can simulate average price when best individual plays with clones
        best_agent = population[0]  # because elites first
        # quick evaluation of average price when best plays with copies
        env_tmp = VectorizedCournotEnv(8, N_PLAYERS_PER_MARKET, ACTIONS)
        states_tmp = env_tmp.reset()
        ps = []
        for t in range(8):
            actions_idx = np.zeros((8, N_PLAYERS_PER_MARKET), dtype=np.int64)
            for e in range(8):
                for p in range(N_PLAYERS_PER_MARKET):
                    s = states_tmp[e, p]
                    actions_idx[e, p] = best_agent.act(s, epsilon=0.0)
            ns, rewards_tmp, _, info_tmp = env_tmp.step(actions_idx)
            ps.append(info_tmp['prices'].mean())
            states_tmp = ns
        price_history.append(float(np.mean(ps)))

    return population, best_history, mean_history, price_history

# -------------------------
# Execução
# -------------------------
if __name__ == "__main__":
    t0 = time.time()
    pop, best_hist, mean_hist, price_hist = run_evolution()
    print(f"\nEvolução completa em {time.time() - t0:.1f}s")

    # Benchmarks analíticos (por firma)
    q_cournot = (a - c) / (b * (N_PLAYERS_PER_MARKET + 1))
    q_collusion = (a - c) / (2 * b * N_PLAYERS_PER_MARKET)
    q_perfect = (a - c) / (b * N_PLAYERS_PER_MARKET)
    P_cournot = a - b * N_PLAYERS_PER_MARKET * q_cournot
    P_collusion = a - b * N_PLAYERS_PER_MARKET * q_collusion
    P_perfect = c

    # Plots fitness
    plt.figure(figsize=(10,4))
    plt.plot(best_hist, label='best fitness por geração')
    plt.plot(mean_hist, label='fitness médio por geração')
    plt.xlabel('Geração')
    plt.ylabel('Lucro médio por firma (fitness)')
    plt.legend()
    plt.grid(True)
    plt.title('Fitness ao longo das gerações')
    plt.show()

    # Price plot
    plt.figure(figsize=(10,4))
    plt.plot(price_hist, label='Preço médio (best) por geração')
    plt.axhline(P_cournot, color='red', linestyle='--', label='P Cournot')
    plt.axhline(P_collusion, color='green', linestyle=':', label='P Colusão')
    plt.axhline(P_perfect, color='blue', linestyle='-.', label='P Concorrência perfeita')
    plt.xlabel('Geração')
    plt.ylabel('Preço médio')
    plt.legend()
    plt.grid(True)
    plt.title('Preço médio (política elite) por geração')
    plt.show()

    # Print política final média do elite
    elite = pop[0]
    # compute preferred action at zero state
    s0 = np.zeros((STATE_DIM,), dtype=np.float32)
    pref_idx = elite.act(s0, epsilon=0)
    print(f"Elite ação preferida (index) {pref_idx} quantidade {ACTIONS[pref_idx]:.3f}")
