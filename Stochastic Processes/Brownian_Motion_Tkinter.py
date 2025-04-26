import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import bisect

class BrownianBridgePath:
    def __init__(self, T):
        self.T = T
        # endpoints of the bridge
        self.points = {0.0: 0.0, T: np.random.randn() * np.sqrt(T)}

    def sample(self, t):
        """Sample B(t) via Brownian bridge if not already stored."""
        if t in self.points:
            return self.points[t]
        times = sorted(self.points)
        i = bisect.bisect_left(times, t)
        t0, t1 = times[i-1], times[i]
        B0, B1 = self.points[t0], self.points[t1]
        # Brownian bridge mean & variance
        mean = (B0 * (t1 - t) + B1 * (t - t0)) / (t1 - t0)
        var  = (t - t0) * (t1 - t) / (t1 - t0)
        Bt   = mean + np.random.randn() * np.sqrt(var)
        self.points[t] = Bt
        return Bt

    def get_region(self, x0, x1, n_points):
        """Return n_points samples of the bridge in [x0, x1]."""
        self.sample(x0)
        self.sample(x1)
        ts = np.linspace(x0, x1, n_points)
        Bs = np.array([self.sample(float(t)) for t in ts])
        return ts, Bs

class InfiniteBrownianPlot:
    def __init__(self, parent, title, n_paths, T, transform=lambda B, t: B, max_total=200_000):
        self.T = T
        self.transform = transform
        self.max_total = max_total
        self.paths = [BrownianBridgePath(T) for _ in range(n_paths)]

        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xlim(0, T)

        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, parent).update()
        self.canvas.draw()

        # initial draw
        self._draw_region(0.0, T)

        # connect zoom callback
        self.ax.callbacks.connect('xlim_changed', self._on_zoom)

    def _draw_region(self, x0, x1):
        """Clear & draw region [x0,x1], sampling proportional to zoom."""
        self.ax.cla()
        self.ax.set_title(self.ax.get_title())
        self.ax.set_xlim(x0, x1)

        # compute desired number of points:
        full_range = self.T
        curr_range = x1 - x0
        zoom_factor = full_range / curr_range
        # width in pixels
        w, _ = self.canvas.get_width_height()
        desired = int(w * zoom_factor)
        n = max(2, min(desired, self.max_total))

        for path in self.paths:
            ts, Bs = path.get_region(x0, x1, n)
            Xs = self.transform(Bs, ts)
            self.ax.scatter(ts, Xs, s=5)

        self.canvas.draw_idle()

    def _on_zoom(self, ax):
        x0, x1 = ax.get_xlim()
        x0, x1 = max(0.0, x0), min(self.T, x1)
        self._draw_region(x0, x1)

def main():
    root = tk.Tk()
    root.title("Browniano com Resolução Dinâmica")
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Exercício 1: padrão
    f1 = ttk.Frame(notebook); notebook.add(f1, text="Exercício 1")
    InfiniteBrownianPlot(f1,
        title="Browniano Padrão (5 trajetórias)",
        n_paths=5,
        T=1.0
    )

    # Exercício 2: aritmético
    x0, mu, sigma = 1.0, 0.05, 0.2
    f2 = ttk.Frame(notebook); notebook.add(f2, text="Exercício 2")
    InfiniteBrownianPlot(f2,
        title="Browniano Aritmético (10 trajetórias)",
        n_paths=10,
        T=20.0,
        transform=lambda B, t: x0 + mu * t + sigma * B
    )

    # Exercício 3: geométrico
    f3 = ttk.Frame(notebook); notebook.add(f3, text="Exercício 3")
    InfiniteBrownianPlot(f3,
        title="Browniano Geométrico (10 trajetórias)",
        n_paths=10,
        T=20.0,
        transform=lambda B, t: x0 * np.exp(mu * t + sigma * B)
    )

    root.mainloop()

if __name__ == "__main__":
    main()
