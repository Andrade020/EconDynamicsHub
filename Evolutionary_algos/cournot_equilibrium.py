import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from joblib import Parallel, delayed

# -------------------------
# Parâmetros globais
# -------------------------
a, b, c = 100.0, 1.0, 20.0
N, M, G, R = 4, 1000, 10000, 5
mutation_rate, mutation_std = 0.05, 2.0
q_max = (a - c) / b

# -------------------------
# Funções aceleradas
# -------------------------
@njit
def profits_numba(q_vector, a, b, c):
    Q = np.sum(q_vector)
    P = a - b * Q
    if P < 0:
        P = 0.0
    return (P - c) * q_vector

@njit(parallel=True)
def simulate_evolution(seed, a, b, c, N, M, G, R, mutation_rate, mutation_std, q_max):
    np.random.seed(seed)
    population = np.random.uniform(0, q_max, size=M)
    fitness = np.zeros(M)

    mean_q_history = np.zeros(G)
    mean_price_history = np.zeros(G)

    for gen in range(G):
        fitness[:] = 0.0

        for _ in range(R):
            # escolha manual sem reposição
            idx = np.random.permutation(M)[:N]
            q_vector = population[idx]
            profit_vector = profits_numba(q_vector, a, b, c)
            for i in range(N):
                fitness[idx[i]] += profit_vector[i]

        mean_q = np.mean(population)
        Q_total = N * mean_q
        P_market = max(a - b * Q_total, 0)
        mean_q_history[gen] = mean_q
        mean_price_history[gen] = P_market

        # torneios paralelos
        for _ in prange(M):
            i = np.random.randint(0, M)
            j = np.random.randint(0, M)
            winner = i if fitness[i] > fitness[j] else j
            loser = j if winner == i else i

            child_q = population[winner]
            if np.random.random() < mutation_rate:
                child_q += np.random.normal(0, mutation_std)
                if child_q < 0:
                    child_q = 0.0
                elif child_q > q_max:
                    child_q = q_max
            population[loser] = child_q

    return mean_q_history, mean_price_history

# -------------------------
# Execução paralela
# -------------------------
n_jobs = 4  # número de núcleos a usar
seeds = np.arange(n_jobs)
results = Parallel(n_jobs=n_jobs)(
    delayed(simulate_evolution)(
        int(seed), a, b, c, N, M, G, R, mutation_rate, mutation_std, q_max
    )
    for seed in seeds
)

# Média das simulações
mean_q_matrix = np.array([r[0] for r in results])
mean_p_matrix = np.array([r[1] for r in results])
mean_q = mean_q_matrix.mean(axis=0)
mean_p = mean_p_matrix.mean(axis=0)

# -------------------------
# Benchmarks
# -------------------------
q_cournot = (a - c) / (b * (N + 1))
q_collusion = (a - c) / (2 * b * N)
q_perfect = (a - c) / (b * N)
P_cournot = a - b * N * q_cournot
P_collusion = a - b * N * q_collusion
P_perfect = a - b * N * q_perfect  # = c

# -------------------------
# Gráficos
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(mean_q, label='Quantidade média (q)')
plt.axhline(q_cournot, color='red', linestyle='--', label='Cournot')
plt.axhline(q_collusion, color='green', linestyle=':', label='Colusão perfeita')
plt.axhline(q_perfect, color='blue', linestyle='-.', label='Concorrência perfeita')
plt.xlabel("Geração")
plt.ylabel("Quantidade média por firma")
plt.title("Evolução das estratégias") # (com Numba + paralelismo)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(mean_p, label='Preço médio')
plt.axhline(P_cournot, color='red', linestyle='--', label='Cournot')
plt.axhline(P_collusion, color='green', linestyle=':', label='Colusão perfeita')
plt.axhline(P_perfect, color='blue', linestyle='-.', label='Concorrência perfeita')
plt.title("Preço médio por geração")
plt.xlabel("Geração")
plt.ylabel("Preço")
plt.legend()
plt.tight_layout()
plt.show()
