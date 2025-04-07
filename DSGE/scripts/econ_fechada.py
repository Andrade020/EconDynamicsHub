import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do modelo
alpha = 0.36        # elasticidade da produção em relação ao capital
beta = 0.96         # fator de desconto
delta = 0.08        # taxa de depreciação
T = 100             # número de períodos a simular

# Cálculo do estado estacionário (steady state)
# Na solução do Euler, o estado estacionário satisfaz: 1 = beta * (alpha * k_ss^(alpha-1) + 1 - delta)
# Portanto, k_ss = [alpha / (1/beta - (1-delta))]^(1/(1-alpha))
k_ss = (alpha / (1/beta - (1-delta)))**(1/(1-alpha))
# Consumo estacionário: usando a restrição de recursos no steady state:
c_ss = k_ss**alpha - delta * k_ss

print("Capital estacionário (k*): {:.3f}".format(k_ss))
print("Consumo estacionário (c*): {:.3f}".format(c_ss))

# Inicialização dos vetores para capital e consumo
k = np.zeros(T)
c = np.zeros(T)

# Condições iniciais (escolha de valores fora do steady state para ver a convergência)
k[0] = 5.447 # 2.0      # capital inicial
c[0] =  1.405 #  0.5      # consumo inicial (valor escolhido arbitrariamente)

# Sistema de equações:
# 1. Restrição de acumulação:  k[t+1] = f(k[t]) + (1-delta)*k[t] - c[t], com f(k)=k^alpha
# 2. Equação de Euler para log utilidade: c[t+1] = beta * (alpha * k[t+1]^(alpha-1) + 1-delta) * c[t]
for t in range(T-1):
    k[t+1] = k[t]**alpha + (1-delta)*k[t] - c[t]
    c[t+1] = beta * (alpha * k[t+1]**(alpha - 1) + 1 - delta) * c[t]

# Vetor tempo para os gráficos
time = np.arange(T)

# --- Gráfico 1: Evolução Temporal de Capital e Consumo ---
plt.figure(figsize=(10,6))
plt.plot(time, k, label="Capital")
plt.plot(time, c, label="Consumo")
plt.axhline(y=k_ss, color='gray', linestyle='--', label="k* (estado estacionário)")
plt.axhline(y=c_ss, color='gray', linestyle=':', label="c* (estado estacionário)")
plt.xlabel("Tempo")
plt.ylabel("Níveis")
plt.title("Evolução Temporal de Capital e Consumo")
plt.legend()
plt.grid(True)
plt.show()

# --- Gráfico 2: Diagrama de Fase (Consumo vs. Capital) ---
plt.figure(figsize=(8,6))
plt.plot(k, c, marker='o')
plt.xlabel("Capital")
plt.ylabel("Consumo")
plt.title("Diagrama de Fase: Consumo vs. Capital")
plt.grid(True)
plt.show()

# --- Gráfico 3: Função de Produção e Linha de Depreciação ---
# Para um conjunto de valores de k, plotamos f(k)=k^alpha e a reta delta*k.
k_vals = np.linspace(0.1, 10, 300)
f_k = k_vals**alpha         # função de produção
dep_line = delta * k_vals     # linha de depreciação

plt.figure(figsize=(8,6))
plt.plot(k_vals, f_k, label="Produção f(k) = k^α")
plt.plot(k_vals, dep_line, label="Depreciação δ·k", linestyle="--")
plt.xlabel("Capital")
plt.ylabel("Produção / Depreciação")
plt.title("Função de Produção e Reta de Depreciação")
plt.legend()
plt.grid(True)
plt.show()
