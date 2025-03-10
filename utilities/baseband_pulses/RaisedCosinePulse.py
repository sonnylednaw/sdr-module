import numpy as np

from utilities.baseband_pulses.BasebandPulse import BasebandPulse
from utilities.types.Params import BasebandParams


class RaisedCosinePulse(BasebandPulse):
    def __init__(self, T_s: float, sps: int, num_samples: int, roll_off: float, amplitude: float=1.0):
        super().__init__(T_s, sps, num_samples, amplitude)
        self.roll_off = roll_off

    @classmethod
    def from_baseband_params(cls, baseband_params: BasebandParams):
        return cls(baseband_params.T_s, baseband_params.sps, baseband_params.num_samps, baseband_params.roll_off, baseband_params.amplitude)

    def generate_pulse(self) -> np.ndarray:
        Ts = self.sps
        t = np.arange(self.num_samples) - (self.num_samples - 1) // 2
        h = np.sinc(t / Ts) * np.cos(np.pi * self.roll_off * t / Ts) / (1 - (2 * self.roll_off * t / Ts) ** 2)
        return h
