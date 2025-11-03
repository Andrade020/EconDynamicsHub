import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parâmetros do ambiente (economia)
# -------------------------
a = 100.0
b = 1.0
c = 20.0
N = 4              # número de firmas/agentes
q_max = (a - c) / b

# -------------------------
# Parâmetros do RL
# -------------------------
K = 21             # número de ações (quantidades discretas)
S = 21             # número de estados (bins de preço)
episodes = 8000     # episódios (cada episódio tem T steps)
T = 20             # passos por episódio (jogo repetido)
alpha = 0.1        # taxa de aprendizado
gamma = 0.95       # desconto
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995

# seeds
#seed = 42
rng = np.random.default_rng()

# -------------------------
# Ações e estados discretos
# -------------------------
actions = np.linspace(0.0, q_max, K)  # q discretos
price_bins = np.linspace(0.0, a, S)    # bins para discretizar o preço (0..a)

def price_to_state(p):
    # retorna índice do bin mais próximo (0..S-1)
    idx = np.digitize(p, price_bins) - 1
    idx = max(0, min(S-1, idx))
    return idx

# -------------------------
# Inicializa Q-tables: um por agente
# Q shape: (N, S, K)
# -------------------------
Q = np.zeros((N, S, K), dtype=np.float64)

# estatísticas
mean_q_history = []   # média de q por episódio (média sobre T e agentes)
mean_p_history = []   # média de preço por episódio
epsilon = epsilon_start

# -------------------------
# Função de lucro
# -------------------------
def compute_price_and_profits(qs):
    Qtotal = np.sum(qs)
    P = max(a - b * Qtotal, 0.0)
    profits = (P - c) * qs
    return P, profits

# -------------------------
# Loop de treinamento
# -------------------------
for ep in range(episodes):
    # estado inicial: assumimos preço inicial = c (ou a/2) para começar
    prev_price = c
    state = price_to_state(prev_price)

    ep_qs = []
    ep_ps = []

    for t in range(T):
        # cada agente escolhe ação por epsilon-greedy
        actions_idx = np.zeros(N, dtype=int)
        for i in range(N):
            if rng.random() < epsilon:
                actions_idx[i] = rng.integers(0, K)
            else:
                # tie-breaking aleatório entre argmax
                maxv = Q[i, state].max()
                candidates = np.where(Q[i, state] == maxv)[0]
                actions_idx[i] = rng.choice(candidates)

        qs = actions[actions_idx]
        P, profits = compute_price_and_profits(qs)
        next_state = price_to_state(P)

        # atualizar Q para cada agente (Q-learning independente)
        for i in range(N):
            a_idx = actions_idx[i]
            reward = profits[i]
            best_next = Q[i, next_state].max()
            td_target = reward + gamma * best_next
            td_error = td_target - Q[i, state, a_idx]
            Q[i, state, a_idx] += alpha * td_error

        # registrar
        ep_qs.append(qs.mean())
        ep_ps.append(P)

        state = next_state

    # estatísticas do episódio
    mean_q_history.append(np.mean(ep_qs))
    mean_p_history.append(np.mean(ep_ps))

    # decai epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

# -------------------------
# Benchmarks analíticos (por firma)
# -------------------------
q_cournot = (a - c) / (b * (N + 1))
q_collusion = (a - c) / (2 * b * N)
q_perfect = (a - c) / (b * N)

P_cournot = a - b * N * q_cournot
P_collusion = a - b * N * q_collusion
P_perfect = c  # a - b*Q_perfect

print("Benchmarks (q por firma):")
print(f"Cournot:   {q_cournot:.3f}")
print(f"Colusão:   {q_collusion:.3f}")
print(f"Perf.:     {q_perfect:.3f}")
print("")

# -------------------------
# Gráficos
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(mean_q_history, label='Quantidade média (episódio)')
plt.axhline(q_cournot, color='red', linestyle='--', label='Cournot')
plt.axhline(q_collusion, color='green', linestyle=':', label='Colusão')
plt.axhline(q_perfect, color='blue', linestyle='-.', label='Concorrência perfeita')
plt.xlabel('Episódio')
plt.ylabel('Quantidade média por firma')
plt.title('Evolução - Q-learning independente (quantidades)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(mean_p_history, label='Preço médio (episódio)')
plt.axhline(P_cournot, color='red', linestyle='--', label='P Cournot')
plt.axhline(P_collusion, color='green', linestyle=':', label='P Colusão')
plt.axhline(P_perfect, color='blue', linestyle='-.', label='P Concorrência perfeita')
plt.xlabel('Episódio')
plt.ylabel('Preço médio')
plt.title('Evolução - Q-learning independente (preço)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Política final (por agente): ação média preferida em cada estado
# -------------------------
policy_actions = Q.argmax(axis=2)  # shape (N, S) indices de ação
# calcular a quantidade média escolhida por agente (média sobre estados)
mean_action_per_agent = (actions[policy_actions].mean(axis=1))
for i in range(N):
    print(f"Agente {i}: ação média (política) = {mean_action_per_agent[i]:.3f}")
