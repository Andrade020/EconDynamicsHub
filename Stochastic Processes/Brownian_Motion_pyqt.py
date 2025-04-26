#!/usr/bin/env python3
"""
Brownian Motion with fixed number of points per view, using PyQtGraph.
Zooming in simply redistributes those points over the new X-interval,
without increasing their total count, and preserves your manual zoom both
in X and Y.
"""

import sys
import bisect
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg

class BrownianBridgePath:
    def __init__(self, T):
        self.T = T
        self.points = {0.0: 0.0, T: np.random.randn() * np.sqrt(T)}

    def sample(self, t):
        if t in self.points:
            return self.points[t]
        times = sorted(self.points)
        i = bisect.bisect_left(times, t)
        t0, t1 = times[i-1], times[i]
        B0, B1 = self.points[t0], self.points[t1]
        mean = (B0 * (t1 - t) + B1 * (t - t0)) / (t1 - t0)
        var  = (t - t0) * (t1 - t) / (t1 - t0)
        Bt   = mean + np.random.randn() * np.sqrt(var)
        self.points[t] = Bt
        return Bt

    def get_region(self, x0, x1, n_points):
        self.sample(x0); self.sample(x1)
        ts = np.linspace(x0, x1, n_points)
        Bs = np.array([self.sample(float(t)) for t in ts])
        return ts, Bs

class BrownianPlot(pg.PlotWidget):
    def __init__(self, title, n_paths, T, transform=lambda B,t: B, total_points=320):
        super().__init__()
        self.title_str = title
        self.T = T
        self.transform = transform
        self.total_points = total_points
        self.paths = [BrownianBridgePath(T) for _ in range(n_paths)]
        self.setTitle(self.title_str)
        vb = self.getViewBox()
        vb.enableAutoRange(x=False, y=False)
        vb.sigRangeChanged.connect(self._on_range_changed)

        # prepare scatter items
        self.scats = []
        for _ in range(n_paths):
            scatter = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(50,150,255,120))
            self.addItem(scatter)
            self.scats.append(scatter)

        # initial draw
        self._update_scatter(0.0, T)

    def _update_scatter(self, x0, x1):
        # keep the current zoom ranges
        vb = self.getViewBox()
        xrg, yrg = vb.viewRange()
        # sample a fixed number of points over [x0,x1]
        n = self.total_points
        for path, scatter in zip(self.paths, self.scats):
            ts, Bs = path.get_region(x0, x1, n)
            Xs = self.transform(Bs, ts)
            scatter.setData(ts, Xs)
        # restore the user’s zoom
        vb.setRange(xRange=xrg, yRange=yrg, padding=0)

    def _on_range_changed(self, vb, ranges):
        (x0, x1), _ = ranges
        x0 = max(0.0, x0); x1 = min(self.T, x1)
        self._update_scatter(x0, x1)

def main():
    app = QtWidgets.QApplication(sys.argv)
    tabs = QtWidgets.QTabWidget()
    tabs.setWindowTitle("Browniano – pontos fixos")

    # Ex1: padrão
    w1 = BrownianPlot("Browniano Padrão (5 trajetórias)", n_paths=5, T=1.0)
    tabs.addTab(w1, "Exercício 1")

    # Ex2: aritmético
    x0, mu, sigma = 1.0, 0.05, 0.2
    transform2 = lambda B, t: x0 + mu*t + sigma*B
    w2 = BrownianPlot("Browniano Aritmético (10 trajetórias)",
                      n_paths=10, T=20.0, transform=transform2)
    tabs.addTab(w2, "Exercício 2")

    # Ex3: geométrico
    transform3 = lambda B, t: x0 * np.exp(mu*t + sigma*B)
    w3 = BrownianPlot("Browniano Geométrico (10 trajetórias)",
                      n_paths=10, T=20.0, transform=transform3)
    tabs.addTab(w3, "Exercício 3")

    tabs.resize(900, 600)
    tabs.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
