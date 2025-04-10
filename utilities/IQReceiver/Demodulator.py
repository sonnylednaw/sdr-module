import numpy as np


class Demodulator:
    def __init__(self):
        pass

    @staticmethod
    def bpsk_decision(received_symbols):
        data_bits = np.zeros(len(received_symbols), dtype=int)
        for i, symbol in enumerate(received_symbols):
            if symbol.real > 0:
                data_bits[i] = 1
            else:
                data_bits[i] = 0
        return data_bits

    @staticmethod
    def qpsk_decision(received_symbols):
        data_bits = np.zeros(len(received_symbols) * 2, dtype=int)
        for i, symbol in enumerate(received_symbols):
            if symbol.real > 0 and symbol.imag > 0:
                data_bits[2*i:2*i+2] = [0, 0]
            elif symbol.real < 0 < symbol.imag:
                data_bits[2*i:2*i+2] = [0, 1]
            elif symbol.real < 0 and symbol.imag < 0:
                data_bits[2*i:2*i+2] = [1, 1]
            else:  # symbol.real > 0 and symbol.imag < 0
                data_bits[2*i:2*i+2] = [1, 0]
        return data_bits