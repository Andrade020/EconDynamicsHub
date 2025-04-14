import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import io
from PIL import Image, ImageTk

# Global variables para a simulação
k_vals = []
c_vals = []
production_vals = []
alpha_global = 0.33  # Será atualizado conforme o parâmetro inserido
T_global = 50  # Número de períodos

# Variável global para cache da imagem das fórmulas
cached_formula_image = None

def simulate():
    global k_vals, c_vals, production_vals, alpha_global, T_global
    try:
        beta = float(entry_beta.get())
        alpha = float(entry_alpha.get())
        delta = float(entry_delta.get())
        k0 = float(entry_k0.get())
        c0 = float(entry_c0.get())
        T = int(entry_T.get())
    except ValueError:
        status_label.config(text="Erro: parâmetros inválidos.", fg="red")
        return

    alpha_global = alpha
    T_global = T
    # Inicializa as listas com os valores iniciais
    k_vals = [k0]
    c_vals = [c0]
    production_vals = [k0**alpha]

    # Loop de simulação para cada período
    for t in range(T):
        k_current = k_vals[-1]
        production = k_current**alpha
        production_vals.append(production)

        # Equação de acumulação: k_{t+1} = f(k_t) + (1-δ)k_t - c_t
        k_next = production + (1-delta)*k_current - c_vals[-1]
        k_next = max(k_next, 0)  # Garante que não haja capital negativo
        k_vals.append(k_next)

        # Atualização de consumo via a condição de Euler:
        # c_{t+1} = β * [α * k_{t+1}^{α-1} + (1-δ)] * c_t
        if k_next > 0:
            c_next = beta * (alpha * (k_next**(alpha-1)) + (1-delta)) * c_vals[-1]
        else:
            c_next = 0
        c_vals.append(c_next)

    # Atualiza o gráfico principal (séries temporais de capital e consumo)
    ax_main.clear()
    t_range = np.arange(T+1)
    ax_main.plot(t_range, k_vals, marker="o", label="Capital")
    ax_main.plot(t_range, c_vals, marker="x", label="Consumo")
    ax_main.set_title("Trajetórias de Capital e Consumo")
    ax_main.set_xlabel("Período")
    ax_main.set_ylabel("Níveis")
    ax_main.legend()
    canvas_main.draw()

    status_label.config(text="Simulação executada com sucesso.", fg="green")

def plot_all_graphs():
    # Cria uma nova janela para exibir os gráficos em um grid 2x2
    top = tk.Toplevel(root)
    top.title("Gráficos Completos")

    fig_all, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Gráfico 1: Séries temporais de Capital e Consumo vs Período
    t_range = np.arange(len(k_vals))
    axs[0,0].plot(t_range, k_vals, marker="o", label="Capital")
    axs[0,0].plot(t_range, c_vals, marker="x", label="Consumo")
    axs[0,0].set_title("Trajetórias de Capital e Consumo")
    axs[0,0].set_xlabel("Período")
    axs[0,0].set_ylabel("Níveis")
    axs[0,0].legend()

    # Gráfico 2: Produção vs Capital
    axs[0,1].plot(k_vals, production_vals, marker="o", linestyle="--", color="purple")
    axs[0,1].set_title("Produção vs Capital")
    axs[0,1].set_xlabel("Capital")
    axs[0,1].set_ylabel("Produção")

    # Gráfico 3: Diagrama de Fase (Capital x Consumo)
    axs[1,0].plot(k_vals, c_vals, marker="o", linestyle="-", color="green")
    axs[1,0].set_title("Diagrama de Fase (Capital x Consumo)")
    axs[1,0].set_xlabel("Capital")
    axs[1,0].set_ylabel("Consumo")
    # Cálculo do estado estacionário:
    try:
        beta = float(entry_beta.get())
        alpha = float(entry_alpha.get())
        delta = float(entry_delta.get())
    except ValueError:
        beta, alpha, delta = 0.96, 0.33, 0.1
    if alpha != 1:
        k_star = ((1/beta - (1-delta)) / alpha)**(1/(alpha-1))
        c_star = k_star**alpha - delta*k_star
        axs[1,0].plot(k_star, c_star, marker="s", color="red", markersize=8, label="Steady State")
        axs[1,0].legend()

    # Gráfico 4: Razão Consumo/Capital vs Período
    ratio = np.array(c_vals) / np.array(k_vals, dtype=float)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)  # Corrige divisões por zero
    axs[1,1].plot(t_range, ratio, marker="o", linestyle="--", color="brown")
    axs[1,1].set_title("Razão Consumo/Capital vs Período")
    axs[1,1].set_xlabel("Período")
    axs[1,1].set_ylabel("Consumo/Capital")

    fig_all.tight_layout()
    canvas_all = FigureCanvasTkAgg(fig_all, master=top)
    canvas_all.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas_all.draw()

def show_formulas():
    global cached_formula_image

    # Cria uma nova janela para exibir as fórmulas do modelo
    top = tk.Toplevel(root)
    top.title("Fórmulas do Modelo")

    if cached_formula_image is None:
        # Cria uma figura com as fórmulas renderizadas em LaTeX (sem \Bigl e \Bigr)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axis("off")  # Oculta os eixos para exibir somente o texto

        formula_text = r"""
Problema de Maximização:

$ \max_{\{c_t,\,k_{t+1}\}} \sum_{t=0}^{\infty} \beta^t \ln(c_t) $

sujeito a:

$ k_{t+1} = k_t^\alpha + (1-\delta)k_t - c_t $

Equação de Euler:

$ c_{t+1} = \beta [\alpha k_{t+1}^{\alpha-1} + (1-\delta)] c_t $

Função de Produção:

$ y_t = k_t^\alpha $

Estado Estacionário:

$ 1 = \beta [\alpha (k^*)^{\alpha-1} + (1-\delta)] $

$ c^* = (k^*)^\alpha - \delta k^* $
"""
        ax.text(0.05, 0.95, formula_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', wrap=True)

        # Salva a figura em um buffer (em formato PNG)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        # Converte o buffer para uma imagem Pillow e depois para PhotoImage
        image = Image.open(buf)
        cached_formula_image = ImageTk.PhotoImage(image)
        plt.close(fig)
        buf.close()

    # Exibe a imagem cacheada em um Label
    label = tk.Label(top, image=cached_formula_image)
    label.image = cached_formula_image  # Previne a coleta de lixo
    label.pack(fill=tk.BOTH, expand=True)

# Configuração da interface principal Tkinter
root = tk.Tk()
root.title("Simulação do Modelo de Equilíbrio Geral Dinâmico")

frame_inputs = ttk.Frame(root)
frame_inputs.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Campos de entrada para os parâmetros
ttk.Label(frame_inputs, text="Beta (desconto):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
entry_beta = ttk.Entry(frame_inputs)
entry_beta.insert(0, "0.96")
entry_beta.grid(row=0, column=1, padx=5, pady=2)

ttk.Label(frame_inputs, text="Alpha (produção):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
entry_alpha = ttk.Entry(frame_inputs)
entry_alpha.insert(0, "0.33")
entry_alpha.grid(row=1, column=1, padx=5, pady=2)

ttk.Label(frame_inputs, text="Delta (depreciação):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
entry_delta = ttk.Entry(frame_inputs)
entry_delta.insert(0, "0.1")
entry_delta.grid(row=2, column=1, padx=5, pady=2)

ttk.Label(frame_inputs, text="Capital inicial (k0):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
entry_k0 = ttk.Entry(frame_inputs)
entry_k0.insert(0, "1.0")
entry_k0.grid(row=3, column=1, padx=5, pady=2)

ttk.Label(frame_inputs, text="Consumo inicial (c0):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
entry_c0 = ttk.Entry(frame_inputs)
entry_c0.insert(0, "0.5")
entry_c0.grid(row=4, column=1, padx=5, pady=2)

ttk.Label(frame_inputs, text="Número de períodos:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
entry_T = ttk.Entry(frame_inputs)
entry_T.insert(0, "50")
entry_T.grid(row=5, column=1, padx=5, pady=2)

# Botões para simulação, gráficos completos e fórmulas
button_simulate = ttk.Button(frame_inputs, text="Rodar Simulação", command=simulate)
button_simulate.grid(row=6, column=0, columnspan=2, pady=10)

button_all_graphs = ttk.Button(frame_inputs, text="Mostrar Gráficos Completos", command=plot_all_graphs)
button_all_graphs.grid(row=7, column=0, columnspan=2, pady=10)

button_formulas = ttk.Button(frame_inputs, text="Exibir Fórmulas do Modelo", command=show_formulas)
button_formulas.grid(row=8, column=0, columnspan=2, pady=10)

status_label = ttk.Label(frame_inputs, text="")
status_label.grid(row=9, column=0, columnspan=2)

# Gráfico principal da simulação (Capital e Consumo vs Período)
fig_main, ax_main = plt.subplots(figsize=(6,4))
canvas_main = FigureCanvasTkAgg(fig_main, master=root)
canvas_main.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas_main.draw()

root.mainloop()
