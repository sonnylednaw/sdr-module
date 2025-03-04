from utilities.baseband_pulses.BasebandPulse import BasebandPulse
import numpy as np

class RectPulse(BasebandPulse):
    def __init__(self, Ts=1 / 2, sps=16, num_samples=101, amplitude=1):
        super().__init__(Ts, sps, num_samples, amplitude)

    def generate_pulse(self):
        h = np.zeros(self.num_samples)
        h[self.num_samples // 2 - self.sps // 2:self.num_samples // 2 + self.sps // 2] = self.amplitude
        return h