"""
cournot_rl_grid.py

Q-learning multiagente para oligopólio Cournot com:
 - heterogeneidade de custos (c_i ~ Normal(c0, cost_sd))
 - entrada/saída endógena (firmas com prejuízo persistente são substituídas)
 - mapa de regimes: varredura sobre gamma (discount) e sigma_a (volatilidade de demanda)
 
Métrica final: posição normalizada entre colusão e Cournot (score em [..])
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import time
import os

# -----------------------------
# Parâmetros econômicos padrão
# -----------------------------
a0 = 100.0        # intercepto da demanda média
b = 1.0           # inclinação da demanda
c0 = 20.0         # custo médio
N = 4             # número de firmas no mercado (por rodada)
q_max = (a0 - c0) / b

# -----------------------------
# Parâmetros RL / simulação
# -----------------------------
K = 21            # número de ações (quantidades discretas)
S = 21            # número de bins de preço (estado)
episodes = 500    # episódios por simulação
T = 15            # passos por episódio
alpha = 0.1       # learning rate
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995

# -----------------------------
# Heterogeneidade / entrada-saída
# -----------------------------
cost_sd = 1.5          # desvio padrão dos custos individuais (heterogeneidade)
exit_window = 10       # janela (episódios) para avaliar lucro médio antes de exit
exit_threshold = 0.0   # se lucro médio < threshold -> sai
entrant_epsilon = 0.9  # epsilon inicial para entrants (exploram mais)
entrant_random_frac = 0.3  # frac da Q-table inicializada aleatoriamente para entrants

# -----------------------------
# Grid para mapa de regimes
# -----------------------------
gamma_list = [0.6, 0.75, 0.9, 0.95]       # descontos a testar
sigma_a_list = [0.0, 0.5, 1.0, 2.0, 4.0]  # volatilidade na interceptação da demanda
n_reps = 6   # repetições por ponto da grade (aumente para robustez)
n_jobs = 4   # ajuste conforme CPU

# -----------------------------
# Ações e estados discretos
# -----------------------------
actions = np.linspace(0.0, q_max, K)
price_bins = np.linspace(0.0, a0 + 4 * max(sigma_a_list), S)  # cover possible price range


def price_to_state(p):
    idx = np.digitize(p, price_bins) - 1
    if idx < 0:
        idx = 0
    elif idx >= S:
        idx = S - 1
    return idx


# -----------------------------
# Benchmarks analíticos (por firma)
# -----------------------------
def benchmarks(a, b, c, N):
    q_cournot = (a - c) / (b * (N + 1))
    q_collusion = (a - c) / (2 * b * N)
    q_perfect = (a - c) / (b * N)
    P_cournot = a - b * N * q_cournot
    P_collusion = a - b * N * q_collusion
    P_perfect = c
    return q_cournot, q_collusion, q_perfect, P_cournot, P_collusion, P_perfect


# -----------------------------
# Funções utilitárias: mercado, entrada/saída
# -----------------------------
def compute_price_and_profits(qs, a_t, b, costs):
    Qtotal = np.sum(qs)
    P = max(a_t - b * Qtotal, 0.0)
    profits = (P - costs) * qs
    return P, profits


# -----------------------------
# Função que roda uma simulação (uma repetição) para um dado (gamma, sigma_a)
# Retorna: score final (pos normalized), história média de q e p (opcional)
# -----------------------------
def run_simulation(seed, gamma, sigma_a, verbose=False):
    rng = np.random.default_rng(seed)

    # Inicializa heterogeneidade de custos por firma (tamanho = N; atualizável em entrants)
    costs = rng.normal(c0, cost_sd, size=N)
    # Clip para evitar custos negativos
    costs = np.maximum(costs, 0.5)

    # Q-tables por firma: shape (N, S, K)
    Q = np.zeros((N, S, K), dtype=np.float64)

    # trackers para avaliar exit: lucros por episódio por firma
    profit_history = [[] for _ in range(N)]
    eps = epsilon_start

    # métricas que queremos retornar
    mean_q_series = []
    mean_p_series = []

    for ep in range(episodes):
        # draw demand shock a_t
        a_t = a0 + rng.normal(0.0, sigma_a)
        if a_t <= costs.min():  # avoid degenerate negative demand
            a_t = max(a0 * 0.1, costs.min() + 1e-3)

        # estado inicial: preço anterior aproximado (use c0 if ep==0)
        prev_price = c0
        state = price_to_state(prev_price)

        ep_qs = []
        ep_ps = []
        ep_profit_per_firm = np.zeros(N)

        for t in range(T):
            # escolhas por epsilon-greedy
            actions_idx = np.zeros(N, dtype=int)
            for i in range(N):
                if rng.random() < eps:
                    actions_idx[i] = rng.integers(0, K)
                else:
                    maxv = Q[i, state].max()
                    candidates = np.where(Q[i, state] == maxv)[0]
                    actions_idx[i] = rng.choice(candidates)

            qs = actions[actions_idx]
            P, profits = compute_price_and_profits(qs, a_t, b, costs)
            next_state = price_to_state(P)

            # Q-learning update (independente)
            for i in range(N):
                a_idx = actions_idx[i]
                reward = profits[i]
                best_next = Q[i, next_state].max()
                td_target = reward + gamma * best_next
                td_error = td_target - Q[i, state, a_idx]
                Q[i, state, a_idx] += alpha * td_error

                ep_profit_per_firm[i] += reward

            ep_qs.append(qs.mean())
            ep_ps.append(P)
            state = next_state

        # final do episódio: média de lucros (por episódio)
        episode_profit_avg = ep_profit_per_firm / T
        for i in range(N):
            profit_history[i].append(episode_profit_avg[i])

        mean_q_series.append(np.mean(ep_qs))
        mean_p_series.append(np.mean(ep_ps))

        # Entradas / saídas: verificar cada firma
        for i in range(N):
            # calc lucro médio na janela
            recent = profit_history[i][-exit_window:]
            if len(recent) >= exit_window:
                mean_recent = np.mean(recent)
                if mean_recent < exit_threshold:
                    # firma sai -> substituída por entrant
                    # entrant gets new cost and new Q-table (mix of copy+random)
                    new_cost = rng.normal(c0, cost_sd)
                    new_cost = max(new_cost, 0.5)
                    costs[i] = new_cost

                    # initialize new Q: random small noise, and some random initialization to encourage exploration
                    # Strategy: partly random, partly copy from a good firm
                    # pick a "parent" - firm with top profit in recent window
                    avg_profits = [np.mean(p[-exit_window:]) if len(p) >= exit_window else np.mean(p) for p in profit_history]
                    parent = int(np.argmax(avg_profits))
                    parent_table = Q[parent]

                    # mix parent and random
                    rand_part = rng.normal(0, 1.0, size=(S, K))
                    new_Q = (1 - entrant_random_frac) * parent_table + entrant_random_frac * rand_part
                    Q[i] = new_Q

                    # reset profit history for entrant
                    profit_history[i] = []
                    # give entrant high exploration for some episodes by temporarily increasing epsilon for that firm
                    # simplest: we just globally raise epsilon a bit for next few episodes (or rely on entrant_epsilon)
                    # implement local reset by warming Q with random values -> effective exploration
                    if verbose:
                        print(f"[ep {ep}] firm {i} exited; replaced (parent {parent}, new_cost {new_cost:.2f})")

        # decay epsilon globally
        eps = max(epsilon_end, eps * epsilon_decay)

    # após treinamento, calcular métrica final (média das últimas 20 episódios)
    final_q = np.mean(mean_q_series[-20:])
    q_cournot, q_collusion, q_perfect, _, _, _ = benchmarks(a0, b, c0, N)

    # normaliza entre colusão e Cournot
    denom = (q_cournot - q_collusion)
    if denom == 0:
        score = np.nan
    else:
        score = (final_q - q_collusion) / denom

    return {
        "score": score,
        "final_q": final_q,
        "mean_q_series": mean_q_series,
        "mean_p_series": mean_p_series,
        "costs_final": costs.copy()
    }


# -----------------------------
# Funcao para executar replicações e retornar média
# -----------------------------
def run_replications_for_point(args):
    gamma, sigma_a, seed_base = args
    results = []
    for r in range(n_reps):
        seed = seed_base + r * 1000
        res = run_simulation(seed, gamma, sigma_a, verbose=False)
        results.append(res["score"])
    # média ignorando NaNs
    arr = np.array(results, dtype=np.float64)
    mean_score = np.nanmean(arr)
    return mean_score


# -----------------------------
# Main: varredura em grade (paralela)
# -----------------------------
def main():
    start = time.time()
    grid = list(itertools.product(gamma_list, sigma_a_list))
    tasks = []
    seed_base = 1234
    for (g, s) in grid:
        tasks.append((g, s, seed_base + int((g*100 + s*10)) ))

    # rodar em paralelo
    print(f"Running grid with {len(grid)} points, each with {n_reps} reps. Using {n_jobs} workers.")
    with Pool(processes=n_jobs) as pool:
        results = pool.map(run_replications_for_point, tasks)

    # reshape results into matrix len(gamma) x len(sigma_a)
    res_matrix = np.array(results).reshape(len(gamma_list), len(sigma_a_list))

    # plot heatmap
    plt.figure(figsize=(8, 5))
    im = plt.imshow(res_matrix, origin='lower', aspect='auto', cmap='bwr', vmin=0.0, vmax=1.5)
    plt.colorbar(im, label='score (0=colusão, 1=Cournot, >1 mais competitivo)')
    plt.xticks(ticks=np.arange(len(sigma_a_list)), labels=sigma_a_list)
    plt.yticks(ticks=np.arange(len(gamma_list)), labels=gamma_list)
    plt.xlabel('sigma_a (volatilidade da demanda)')
    plt.ylabel('gamma (discount)')
    plt.title('Mapa de regimes: score médio final (higher = mais competitivo)')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/regime_map_score.png", dpi=200)
    plt.show()

    end = time.time()
    print(f"Done. Time elapsed: {end - start:.1f}s. Heatmap saved to results/regime_map_score.png")


if __name__ == "__main__":
    main()
