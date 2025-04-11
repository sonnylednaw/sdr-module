import numpy as np


class Demodulator:
    def __init__(self):
        pass

    @staticmethod
    def bpsk_decision(received_symbols):

        print("Aufgabe 28: Demodulation BPSK")
        data_bits = np.zeros(len(received_symbols), dtype=int)

        # Implement BPSK demodulation

        return data_bits

    @staticmethod
    def qpsk_decision(received_symbols):
        data_bits = np.zeros(len(received_symbols) * 2, dtype=int)

        print("Aufgabe 28: Demodulation QPSK")

        # Implement QPSK demodulation

        return data_bits