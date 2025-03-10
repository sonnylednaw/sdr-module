from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class BasebandPulse(ABC):
    def __init__(self, Ts=1/2, sps=16, num_samples=101, amplitude=1):
        self.Ts = Ts
        self.sps = sps
        self.num_samples = num_samples
        self.amplitude = amplitude
        self.T_sample = self.Ts / self.sps

    def calculate_pulse_parameters(self):
        h_pulse_form = self.generate_pulse()
        t = np.arange(len(h_pulse_form)) * self.T_sample
        baseband_spectrum = np.fft.fftshift(np.fft.fft(h_pulse_form))
        f = np.fft.fftshift(np.fft.fftfreq(len(h_pulse_form), d=self.T_sample))
        return h_pulse_form, t, baseband_spectrum, f

    def plot(self):
        h_pulse_form, t, baseband_spectrum, f = self.calculate_pulse_parameters()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot time-domain representation
        ax1.plot(t, h_pulse_form, '.')
        ax1.plot(t, h_pulse_form, '-', alpha=0.25)
        ax1.set_xlabel(r'$t$ [s]')
        ax1.set_ylabel(r'$h[n]$')
        ax1.set_title(r"Basisband Impuls $h_{\mathrm{Pulse\,Form}}[n]$ with $T_s = $" + f"{self.Ts}")
        ax1.grid(True)
        ax1.legend(["Samples", "Interpoliert"])

        # Plot frequency-domain representation
        ax2.plot(f, 10 * np.log10(np.abs(baseband_spectrum) ** 2))
        ax2.set_title(
            r"Spektrum des Basisbandimpulses $\mathfrak{F} \{ h_{\mathrm{Pulse\,Form}}[n] \}$ mit $T_s = $" + f"{self.Ts}")
        ax2.set_xlabel(r'$f$ [Hz]')
        ax2.set_ylabel(r'$|H(f)|$ [dB]')
        ax2.grid(True)
        max_value = np.max(10 * np.log10(np.abs(baseband_spectrum) ** 2))
        ax2.set_ylim(max_value - 100, 1.25 * max_value)

        plt.tight_layout()
        plt.show()

    @abstractmethod
    def generate_pulse(self) -> np.ndarray:
        pass
