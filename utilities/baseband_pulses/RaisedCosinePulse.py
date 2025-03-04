import numpy as np

from utilities.baseband_pulses.BasebandPulse import BasebandPulse


class RaisedCosinePulse(BasebandPulse):
    def __init__(self, Ts=1/2, sps=16, num_samples=101, amplitude=1, roll_off=0.35):
        super().__init__(Ts, sps, num_samples, amplitude)
        self.roll_off = roll_off

    def generate_pulse(self) -> np.ndarray:
        Ts = self.sps
        t = np.arange(self.num_samples) - (self.num_samples - 1) // 2
        h = np.sinc(t / Ts) * np.cos(np.pi * self.roll_off * t / Ts) / (1 - (2 * self.roll_off * t / Ts) ** 2)
        return h