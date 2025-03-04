import numpy as np

from utilities.IQSender.Modulator import Modulator
from utilities.baseband_pulses.RaisedCosinePulse import RaisedCosinePulse
from utilities.baseband_pulses.RectPulse import RectPulse
from utilities.enums.BasebandPulseForm import BasebandPulseForm
from utilities.enums.ModulationCodingScheme import ModulationCodingScheme


# TODO:
#  -Frame Struktur Parameter im Konstruktor hinzufügen
#  -Multiplexing Methode hinzufügen
#  -Synchronisationssequenz als Enum und Methode hinzufügen (Eine Klasse mit Sequenzen und Methoden) die dann hier verwenden
#  - Klasse für Erzeugung von SigMf erstellen
#  - Frame Struktur und alle Parameter am Ende irgendwie plotten und anzeigen


class IQSender:
    def __init__(self, bits: np.array, baseband_pulse_form: BasebandPulseForm, modulation_scheme: ModulationCodingScheme, sps: int = 16):
        self.bits = bits
        self.baseband_pulse = baseband_pulse_form
        self.modulation_scheme = modulation_scheme
        self.sps = sps
        self.h: np.array
        self.modulation_symbols: np.ndarray[np.complex64]
        self.create_baseband_pulse()

    def create_baseband_pulse(self, Ts=1/2, sps=16, num_samples=101, amplitude=1, roll_off=0.35):
        if self.baseband_pulse == BasebandPulseForm.RECT:
            self.h = RectPulse(Ts, sps, num_samples, amplitude).generate_pulse()
        elif self.baseband_pulse == BasebandPulseForm.RAISED_COSINE:
            self.h = RaisedCosinePulse(Ts, sps, num_samples, amplitude, roll_off).generate_pulse()
        else:
            raise ValueError("Unknown baseband pulse form")

    def modulate(self):
        if self.modulation_scheme == ModulationCodingScheme.QPSK:
            self.modulation_symbols = Modulator.qpsk_modulation(self.bits)
        elif self.modulation_scheme == ModulationCodingScheme.BPSK:
            self.modulation_symbols = Modulator.qpsk_modulation(self.bits)
        else:
            raise ValueError("Unknown modulation scheme")






