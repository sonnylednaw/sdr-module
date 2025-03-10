import numpy as np


class Signals:

    @staticmethod
    def delta_distribution(t, tau, amplitude=1.0, width=1e-9):
        dt = t[1] - t[0]
        dirac = np.zeros_like(t)
        idx = np.argmin(np.abs(t - tau))
        if width <= dt:
            dirac[idx] = 1
        else:
            half_width = int(width / (2 * dt))
            start = max(0, idx - half_width)
            end = min(len(t), idx + half_width + 1)
            dirac[start:end] = 1
        return amplitude * dirac