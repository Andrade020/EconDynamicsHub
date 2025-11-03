"""
Híbrido Evolutivo-RL para Oligopólio de Cournot - Versão Pro
Implementa:
1. Heterogeneidade de custos marginais
2. Tracking de colusão implícita
3. Robustness checks com múltiplos seeds

Requisitos:
    pip install numpy torch matplotlib joblib scipy

Rode:
    python hybrid_evo_rl_cournot_pro.py
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
from scipy import stats

# -----------------------
# Configuração de Seeds
# -----------------------
BASE_SEED = 12345
N_RUNS = 5  # Número de runs para robustness check

def set_seed(seed):
    """Define seed para reprodutibilidade"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

device = torch.device('cpu')

# -----------------------
# Parâmetros econômicos
# -----------------------
a = 100.0
b = 1.0
c_mean = 20.0  # Custo marginal médio
c_std = 2.5    # Desvio padrão dos custos (heterogeneidade)
N_PLAYERS_PER_MARKET = 4
q_max = (a - c_mean) / b

# -----------------------
# Ações discretas
# -----------------------
K = 31
ACTIONS = np.linspace(0.0, q_max, K)

# -----------------------
# Parâmetros de RL
# -----------------------
STATE_DIM = 4  # [price_prev, own_q_prev, avg_others_prev, own_cost_normalized]
HIDDEN = 64
LR = 5e-4
GAMMA = 0.97
BATCH_SIZE = 64
REPLAY_CAP = 20000
MIN_REPLAY = 400
TARGET_UPDATE = 500
TRAIN_EVERY = 4

# -----------------------
# Parâmetros evolutivos
# -----------------------
POP_SIZE = 40
GENERATIONS = 30
LIFE_STEPS = 2000
N_ENVS = 12
EVAL_EPISODES = 12
TOURNAMENT = 3
ELITISM = 2
MUTATION_STD = 0.02
NEW_RANDOM_PROB = 0.02

# -----------------------
# Parâmetros de tracking de colusão
# -----------------------
COLLUSION_WINDOW = 3  # Gerações consecutivas para detectar colusão
COLLUSION_THRESHOLD = 0.85  # Q_total < threshold * Q_Cournot indica colusão

# -----------------------
# Utilitários
# -----------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def compute_benchmarks(costs):
    """Calcula benchmarks analíticos dado vetor de custos"""
    c_avg = np.mean(costs)
    q_cournot = (a - c_avg) / (b * (N_PLAYERS_PER_MARKET + 1))
    q_collusion = (a - c_avg) / (2 * b * N_PLAYERS_PER_MARKET)
    q_perfect = (a - c_avg) / (b * N_PLAYERS_PER_MARKET)
    
    P_cournot = a - b * N_PLAYERS_PER_MARKET * q_cournot
    P_collusion = a - b * N_PLAYERS_PER_MARKET * q_collusion
    P_perfect = c_avg
    
    return {
        'q_cournot': q_cournot,
        'q_collusion': q_collusion,
        'q_perfect': q_perfect,
        'P_cournot': P_cournot,
        'P_collusion': P_collusion,
        'P_perfect': P_perfect
    }

def compute_price_and_profits(qs, costs):
    """
    qs: np.array shape (n_players,)
    costs: np.array shape (n_players,)
    """
    Q = float(np.sum(qs))
    P = max(a - b * Q, 0.0)
    profits = (P - costs) * qs
    return P, profits

def normalize_state(p, own_q_prev, avg_others_prev, own_cost):
    """State normalizado incluindo custo próprio"""
    return np.array([
        p / a,
        own_q_prev / q_max if q_max > 0 else 0.0,
        avg_others_prev / q_max if q_max > 0 else 0.0,
        (own_cost - c_mean) / (c_std * 3)  # Normalizado em torno de [-1, 1]
    ], dtype=np.float32)

# -----------------------
# Ambiente vetorizado com custos heterogêneos
# -----------------------
class VectorizedCournotEnv:
    def __init__(self, n_envs, n_players, actions_array, costs_per_player):
        """
        costs_per_player: array (n_players,) com custos marginais de cada firma
        """
        self.n_envs = n_envs
        self.n_players = n_players
        self.actions_array = actions_array
        self.costs = costs_per_player  # Custos fixos por firma
        self.K = len(actions_array)
        self.reset()

    def reset(self):
        self.price_prev = np.full(self.n_envs, c_mean, dtype=np.float32)
        self.own_prev = np.zeros((self.n_envs, self.n_players), dtype=np.float32)
        self.avg_others_prev = np.zeros((self.n_envs, self.n_players), dtype=np.float32)
        
        states = np.zeros((self.n_envs, self.n_players, STATE_DIM), dtype=np.float32)
        for e in range(self.n_envs):
            for i in range(self.n_players):
                states[e, i] = normalize_state(self.price_prev[e], 0.0, 0.0, self.costs[i])
        self.t = 0
        return states

    def step(self, actions_idx, participants):
        """
        actions_idx: (n_envs, n_players) indices
        participants: (n_envs, n_players) agent indices
        """
        n = self.n_envs
        rewards = np.zeros((n, self.n_players), dtype=np.float32)
        next_states = np.zeros((n, self.n_players, STATE_DIM), dtype=np.float32)
        prices = np.zeros(n, dtype=np.float32)

        for e in range(n):
            qs = self.actions_array[actions_idx[e]]
            # Custos das firmas participantes neste mercado
            market_costs = self.costs[participants[e]]
            P, profits = compute_price_and_profits(qs, market_costs)
            prices[e] = P
            rewards[e, :] = profits.astype(np.float32)
            avg_others = (qs.sum() - qs) / max(1, (self.n_players - 1))
            
            for i in range(self.n_players):
                firm_cost = market_costs[i]
                next_states[e, i] = normalize_state(P, qs[i], avg_others[i], firm_cost)
            
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
# Agente evolutivo com DQN
# -----------------------
class EvoAgent:
    def __init__(self, idx, marginal_cost, state_dim=STATE_DIM, n_actions=K, lr=LR):
        self.id = idx
        self.marginal_cost = marginal_cost  # Custo marginal heterogêneo
        self.n_actions = n_actions
        self.q_net = QNet(state_dim, n_actions).to(device)
        self.target_net = QNet(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(REPLAY_CAP)
        self.learn_steps = 0
        self.participation = 0

    def act(self, state, epsilon=0.1):
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
        child = EvoAgent(new_id, self.marginal_cost)
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
# Avaliação com custos heterogêneos
# -----------------------
def evaluate_agent_full(agent, costs_vector, n_envs=12, steps=12):
    """Avalia agente quando todos os jogadores usam a mesma política mas com custos diferentes"""
    # Cria um subconjunto de custos para avaliação
    n_firms = min(N_PLAYERS_PER_MARKET, len(costs_vector))
    eval_costs = costs_vector[:n_firms]
    
    env = VectorizedCournotEnv(n_envs, n_firms, ACTIONS, eval_costs)
    states = env.reset()
    total_profits = 0.0
    total_q = 0.0
    total_price = 0.0
    count = 0
    
    # Participants: índices dos jogadores com custos diferentes
    participants = np.tile(np.arange(n_firms), (n_envs, 1))
    
    for _ in range(steps):
        actions_idx = np.zeros((n_envs, n_firms), dtype=np.int64)
        for e in range(n_envs):
            for p in range(n_firms):
                s = states[e, p]
                actions_idx[e, p] = agent.act(s, epsilon=0.0)
        ns, rewards, dones, info = env.step(actions_idx, participants)
        total_profits += rewards.sum()
        total_q += info['q'].mean()
        total_price += info['prices'].mean()
        count += (n_envs * n_firms)
        states = ns
    
    avg_profit = total_profits / count
    avg_q = total_q / steps
    avg_price = total_price / steps
    return avg_profit, float(avg_q), float(avg_price)

# -----------------------
# Detector de colusão
# -----------------------
class CollusionTracker:
    def __init__(self, window=COLLUSION_WINDOW, threshold=COLLUSION_THRESHOLD):
        self.window = window
        self.threshold = threshold
        self.history = deque(maxlen=window)
        self.collusion_episodes = []
    
    def update(self, q_mean, q_cournot, generation):
        """Atualiza histórico e detecta colusão"""
        q_total_market = q_mean * N_PLAYERS_PER_MARKET
        q_cournot_total = q_cournot * N_PLAYERS_PER_MARKET
        
        is_collusive = q_total_market < (self.threshold * q_cournot_total)
        self.history.append(is_collusive)
        
        # Detecta colusão se todas as últimas 'window' gerações foram colusivas
        if len(self.history) == self.window and all(self.history):
            self.collusion_episodes.append(generation)
            return True
        return False
    
    def get_collusion_rate(self):
        """Retorna % de gerações com comportamento colusivo"""
        if len(self.history) == 0:
            return 0.0
        return sum(self.history) / len(self.history)

# -----------------------
# Loop principal
# -----------------------
def run_hybrid_evo(seed=BASE_SEED, verbose=True):
    set_seed(seed)
    
    # Gera custos heterogêneos para a população (fixos durante toda a simulação)
    costs_population = np.random.uniform(c_mean - c_std, c_mean + c_std, POP_SIZE)
    benchmarks = compute_benchmarks(costs_population)
    
    # Init população com custos heterogêneos
    population = [EvoAgent(i, costs_population[i]) for i in range(POP_SIZE)]
    
    tracker = CollusionTracker()
    mean_q_history = []
    mean_p_history = []
    fitness_history = []
    collusion_rate_history = []
    
    start = time.time()

    for gen in range(GENERATIONS):
        t0 = time.time()
        if verbose:
            print(f"\n--- Geração {gen+1}/{GENERATIONS} (seed {seed}) ---")
        
        # LIFE phase
        # Usamos toda a população de custos para o ambiente
        env = VectorizedCournotEnv(N_ENVS, N_PLAYERS_PER_MARKET, ACTIONS, costs_population)
        states = env.reset()
        
        for step in range(LIFE_STEPS):
            participants = np.zeros((N_ENVS, N_PLAYERS_PER_MARKET), dtype=int)
            for e in range(N_ENVS):
                participants[e] = np.random.choice(len(population), size=N_PLAYERS_PER_MARKET, replace=False)

            actions_idx = np.zeros((N_ENVS, N_PLAYERS_PER_MARKET), dtype=np.int64)
            for e in range(N_ENVS):
                for p in range(N_PLAYERS_PER_MARKET):
                    agent_idx = participants[e, p]
                    agent = population[agent_idx]
                    s = states[e, p]
                    eps = max(0.02, 0.6 * (1.0 - step / LIFE_STEPS))
                    actions_idx[e, p] = agent.act(s, epsilon=eps)

            next_states, rewards, dones, info = env.step(actions_idx, participants)

            for e in range(N_ENVS):
                for p in range(N_PLAYERS_PER_MARKET):
                    agent_idx = participants[e, p]
                    agent = population[agent_idx]
                    s = states[e, p]
                    a = int(actions_idx[e, p])
                    r = float(rewards[e, p])
                    ns = next_states[e, p]
                    agent.store(s, a, r, ns, False)
                    if agent.participation % TRAIN_EVERY == 0:
                        _ = agent.sample_and_learn()

            states = next_states

            if verbose and (step+1) % (LIFE_STEPS // 4) == 0:
                print(f"  life step {step+1}/{LIFE_STEPS} ({time.time()-t0:.1f}s)")

        # Avaliação paralela
        if verbose:
            print("  Avaliando população...")
        results = Parallel(n_jobs=min(12, POP_SIZE))(
            delayed(evaluate_agent_full)(population[i], costs_population, n_envs=12, steps=12) 
            for i in range(len(population))
        )
        
        profits = np.array([r[0] for r in results], dtype=float)
        qs = np.array([r[1] for r in results], dtype=float)
        ps = np.array([r[2] for r in results], dtype=float)

        best_idx = int(np.argmax(profits))
        mean_fit = float(np.mean(profits))
        best_fit = float(profits[best_idx])
        
        fitness_history.append(mean_fit)
        mean_q_history.append(float(qs[best_idx]))
        mean_p_history.append(float(ps[best_idx]))
        
        # Tracking de colusão
        is_collusive = tracker.update(qs[best_idx], benchmarks['q_cournot'], gen+1)
        collusion_rate = tracker.get_collusion_rate()
        collusion_rate_history.append(collusion_rate)
        
        if verbose:
            print(f"  Gen {gen+1}: best_profit={best_fit:.3f} mean_profit={mean_fit:.3f}")
            print(f"  Best agent: q={qs[best_idx]:.3f} p={ps[best_idx]:.3f} cost={costs_population[best_idx]:.2f}")
            print(f"  Collusion rate: {collusion_rate:.1%} {'[COLLUSIVE PATTERN DETECTED]' if is_collusive else ''}")
            print(f"  Benchmarks: q_Cournot={benchmarks['q_cournot']:.3f} q_Collusion={benchmarks['q_collusion']:.3f}")

        # Seleção evolutiva
        sorted_idx = np.argsort(-profits)
        new_pop = []
        
        for k in range(ELITISM):
            idx = sorted_idx[k]
            child = population[idx].clone(new_id=k)
            new_pop.append(child)

        next_id = ELITISM
        while len(new_pop) < POP_SIZE:
            cand = np.random.choice(range(len(population)), size=TOURNAMENT, replace=False)
            cand_best = cand[int(np.argmax(profits[cand]))]
            if random.random() < NEW_RANDOM_PROB:
                ind = EvoAgent(next_id, costs_population[next_id])
            else:
                ind = population[cand_best].clone(new_id=next_id)
                ind.mutate(std=MUTATION_STD)
            new_pop.append(ind)
            next_id += 1

        population = new_pop

    if verbose:
        print(f"\n✓ Run completo em {time.time()-start:.1f}s")
        print(f"Episódios de colusão detectados: {tracker.collusion_episodes}")
    
    return {
        'fitness': fitness_history,
        'q': mean_q_history,
        'p': mean_p_history,
        'collusion_rate': collusion_rate_history,
        'collusion_episodes': tracker.collusion_episodes,
        'benchmarks': benchmarks,
        'costs': costs_population  # Retorna todos os custos da população
    }

# -----------------------
# Robustness check com múltiplos seeds
# -----------------------
def robustness_analysis(n_runs=N_RUNS):
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS CHECK: {n_runs} runs independentes")
    print(f"{'='*60}")
    
    all_results = []
    for i in range(n_runs):
        seed = BASE_SEED + i * 1000
        print(f"\n>>> RUN {i+1}/{n_runs} (seed={seed})")
        result = run_hybrid_evo(seed=seed, verbose=(i==0))  # verbose apenas no primeiro
        all_results.append(result)
    
    return all_results

# -----------------------
# Plotagem com intervalos de confiança
# -----------------------
def plot_results_with_ci(all_results):
    n_gens = len(all_results[0]['q'])
    gens = np.arange(1, n_gens + 1)
    
    # Agregar dados de todos os runs
    all_q = np.array([r['q'] for r in all_results])
    all_p = np.array([r['p'] for r in all_results])
    all_fitness = np.array([r['fitness'] for r in all_results])
    all_collusion = np.array([r['collusion_rate'] for r in all_results])
    
    # Estatísticas
    q_mean = np.mean(all_q, axis=0)
    q_std = np.std(all_q, axis=0)
    q_ci = 1.96 * q_std / np.sqrt(len(all_results))  # 95% CI
    
    p_mean = np.mean(all_p, axis=0)
    p_std = np.std(all_p, axis=0)
    p_ci = 1.96 * p_std / np.sqrt(len(all_results))
    
    fit_mean = np.mean(all_fitness, axis=0)
    fit_ci = 1.96 * np.std(all_fitness, axis=0) / np.sqrt(len(all_results))
    
    col_mean = np.mean(all_collusion, axis=0)
    
    # Benchmarks (média dos runs)
    benchmarks = all_results[0]['benchmarks']
    
    # Plot 1: Quantidades
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gens, q_mean, 'o-', color='navy', linewidth=2, label='Q médio (elite)')
    plt.fill_between(gens, q_mean - q_ci, q_mean + q_ci, alpha=0.3, color='navy', label='95% CI')
    plt.axhline(benchmarks['q_cournot'], color='red', linestyle='--', linewidth=2, label='Q Cournot')
    plt.axhline(benchmarks['q_collusion'], color='green', linestyle=':', linewidth=2, label='Q Colusão')
    plt.axhline(benchmarks['q_perfect'], color='blue', linestyle='-.', linewidth=2, label='Q Competitivo')
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Quantidade média por firma', fontsize=12)
    plt.title(f'Evolução da Quantidade (n={len(all_results)} runs)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Taxa de colusão
    plt.subplot(1, 2, 2)
    plt.plot(gens, col_mean * 100, 'o-', color='darkred', linewidth=2, label='Taxa de colusão')
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Taxa de colusão (%)', fontsize=12)
    plt.title('Colusão Implícita ao Longo do Tempo', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Preços
    plt.figure(figsize=(10, 5))
    plt.plot(gens, p_mean, 'o-', color='darkgreen', linewidth=2, label='P médio (elite)')
    plt.fill_between(gens, p_mean - p_ci, p_mean + p_ci, alpha=0.3, color='darkgreen', label='95% CI')
    plt.axhline(benchmarks['P_cournot'], color='red', linestyle='--', linewidth=2, label='P Cournot')
    plt.axhline(benchmarks['P_collusion'], color='green', linestyle=':', linewidth=2, label='P Colusão')
    plt.axhline(benchmarks['P_perfect'], color='blue', linestyle='-.', linewidth=2, label='P Competitivo (c)')
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Preço médio', fontsize=12)
    plt.title(f'Evolução do Preço (n={len(all_results)} runs)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Fitness
    plt.figure(figsize=(10, 5))
    plt.plot(gens, fit_mean, 'o-', color='purple', linewidth=2, label='Fitness médio')
    plt.fill_between(gens, fit_mean - fit_ci, fit_mean + fit_ci, alpha=0.3, color='purple', label='95% CI')
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Lucro médio por firma', fontsize=12)
    plt.title(f'Evolução do Fitness (n={len(all_results)} runs)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Análise estatística final
    print("\n" + "="*60)
    print("ANÁLISE ESTATÍSTICA FINAL")
    print("="*60)
    
    final_q = all_q[:, -1]
    final_p = all_p[:, -1]
    final_fitness = all_fitness[:, -1]
    
    print(f"\nQuantidade final (última geração):")
    print(f"  Média: {np.mean(final_q):.3f} ± {np.std(final_q):.3f}")
    print(f"  Q Cournot: {benchmarks['q_cournot']:.3f}")
    print(f"  Q Colusão: {benchmarks['q_collusion']:.3f}")
    
    # Teste t: Q final vs Q Cournot
    t_stat, p_val = stats.ttest_1samp(final_q, benchmarks['q_cournot'])
    print(f"  Teste t (H0: Q_final = Q_Cournot): t={t_stat:.3f}, p={p_val:.4f}")
    
    print(f"\nPreço final:")
    print(f"  Média: {np.mean(final_p):.3f} ± {np.std(final_p):.3f}")
    print(f"  P Cournot: {benchmarks['P_cournot']:.3f}")
    print(f"  P Colusão: {benchmarks['P_collusion']:.3f}")
    
    print(f"\nFitness final:")
    print(f"  Média: {np.mean(final_fitness):.3f} ± {np.std(final_fitness):.3f}")
    
    # Análise de colusão
    collusion_detected = sum(len(r['collusion_episodes']) > 0 for r in all_results)
    print(f"\nColusão implícita:")
    print(f"  Detectada em {collusion_detected}/{len(all_results)} runs ({100*collusion_detected/len(all_results):.1f}%)")
    print(f"  Taxa média final: {np.mean(all_collusion[:, -1])*100:.1f}%")
    
    # Heterogeneidade de custos
    print(f"\nHeterogeneidade de custos (primeiro run):")
    costs = all_results[0]['costs']
    print(f"  Custos marginais: [{', '.join([f'{c:.2f}' for c in costs])}]")
    print(f"  Média: {np.mean(costs):.2f}, Desvio: {np.std(costs):.2f}")
    print(f"  Min: {np.min(costs):.2f}, Max: {np.max(costs):.2f}")

# -----------------------
# Execução principal
# -----------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID EVO-RL COURNOT - VERSÃO PRO")
    print("="*60)
    print("\nMelhorias implementadas:")
    print("✓ 1. Heterogeneidade de custos marginais")
    print("✓ 2. Tracking de colusão implícita")
    print("✓ 3. Robustness checks com múltiplos seeds")
    print("="*60)
    
    t_start = time.time()
    
    # Executa robustness analysis
    all_results = robustness_analysis(n_runs=N_RUNS)
    
    print(f"\n✓ Todas as simulações completas em {time.time()-t_start:.1f}s")
    
    # Plota resultados com intervalos de confiança
    plot_results_with_ci(all_results)
    
    print("\n" + "="*60)
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)