import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class BrownianBridgePath:
    def __init__(self, T):
        self.T = T
        self.points = {0.0: 0.0, T: np.random.randn() * np.sqrt(T)}
        self.times = [0.0, T]

    def refine_region(self, t0, t1, max_points):
        dt_target = (t1 - t0) / max_points
        # Refine until no segment in [t0,t1] exceeds dt_target
        while True:
            to_add = []
            for a, b in zip(self.times[:-1], self.times[1:]):
                if b <= t0 or a >= t1:
                    continue
                if (b - a) > dt_target:
                    m = (a + b) / 2
                    if m not in self.points:
                        Ba, Bb = self.points[a], self.points[b]
                        var = (b - a) / 4
                        Bm = np.random.randn() * np.sqrt(var) + (Ba + Bb) / 2
                        to_add.append((m, Bm))
            if not to_add:
                break
            for m, val in to_add:
                self.points[m] = val
            self.times = sorted(self.points.keys())

    def segment(self, t0, t1, max_points):
        self.refine_region(t0, t1, max_points)
        ts = [t for t in self.times if t0 <= t <= t1]
        vs = [self.points[t] for t in ts]
        if len(ts) > max_points:
            step = len(ts) // max_points
            ts = ts[::step]
            vs = vs[::step]
        return np.array(ts), np.array(vs)

class InfiniteBrownianPlot:
    def __init__(self, parent, title, n_paths, T, transform=lambda B, t: B, max_points=500):
        self.T = T
        self.max_points = max_points
        self.transform = transform
        self.paths = [BrownianBridgePath(T) for _ in range(n_paths)]

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlim(0, T)

        # initial draw: full interval
        for p in self.paths:
            ts, Bs = p.segment(0.0, T, self.max_points)
            Xs = transform(Bs, ts)
            ax.scatter(ts, Xs, s=5)

        ax.callbacks.connect('xlim_changed', self.on_xlim_changed)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.draw()

    def on_xlim_changed(self, ax):
        x0, x1 = ax.get_xlim()
        ax.cla()
        ax.set_title(ax.get_title())
        ax.set_xlim(x0, x1)
        for p in self.paths:
            ts, Bs = p.segment(max(x0, 0.0), min(x1, self.T), self.max_points)
            Xs = self.transform(Bs, ts)
            ax.scatter(ts, Xs, s=5)
        ax.figure.canvas.draw_idle()

def main():
    root = tk.Tk()
    root.title("Simulações de Browniano - Resolução Infinita")
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Exercício 1: Browniano padrão
    frame1 = ttk.Frame(notebook)
    notebook.add(frame1, text="Exercício 1")
    InfiniteBrownianPlot(frame1,
                         title="Browniano Padrão (5 trajetórias)",
                         n_paths=5,
                         T=1.0)

    # Exercício 2: Browniano aritmético
    x0, mu, sigma = 1.0, 0.05, 0.2
    frame2 = ttk.Frame(notebook)
    notebook.add(frame2, text="Exercício 2")
    InfiniteBrownianPlot(frame2,
                         title="Browniano Aritmético (10 trajetórias)",
                         n_paths=10,
                         T=20.0,
                         transform=lambda B, t: x0 + mu * t + sigma * B)

    # Exercício 3: Browniano geométrico
    frame3 = ttk.Frame(notebook)
    notebook.add(frame3, text="Exercício 3")
    InfiniteBrownianPlot(frame3,
                         title="Browniano Geométrico (10 trajetórias)",
                         n_paths=10,
                         T=20.0,
                         transform=lambda B, t: x0 * np.exp(mu * t + sigma * B))

    root.mainloop()

if __name__ == "__main__":
    main()
