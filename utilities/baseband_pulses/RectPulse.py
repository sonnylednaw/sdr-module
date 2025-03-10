from utilities.baseband_pulses.BasebandPulse import BasebandPulse
import numpy as np

from utilities.types.Params import BasebandParams


class RectPulse(BasebandPulse):
    def __init__(self, T_s: float, sps: int, num_samples: int, amplitude: float=1.0):
        super().__init__(T_s, sps, num_samples, amplitude)

    @classmethod
    def from_baseband_params(cls, baseband_params: BasebandParams):
        return cls(baseband_params.T_s, baseband_params.sps, baseband_params.num_samps, baseband_params.amplitude)

    def generate_pulse(self):
        h = np.zeros(self.num_samples)
        h[self.num_samples // 2 - self.sps // 2:self.num_samples // 2 + self.sps // 2] = self.amplitude
        return h