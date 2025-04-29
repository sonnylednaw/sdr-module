from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import _32Bit

from utilities.enums.ModulationCodingScheme import ModulationCodingScheme


class Modulator:
    def __init__(self):
        pass

    @staticmethod
    def bpsk_modulation(data_bits) -> np.ndarray[Any, np.dtype[np.int8]]:
        """!
        @brief BPSK Modulation
        @param data_bits Input data bits
        @return Modulated BPSK symbols as numpy array
        """

        print("Aufgabe 3.1: BPSK Modulation")
        modulation_data_symbols = []

        # Implement BPSK modulation

        return np.asarray(modulation_data_symbols)

    @staticmethod
    def qpsk_modulation(data_bits) -> np.ndarray[Any, np.dtype[np.complexfloating[_32Bit, _32Bit]]]:
        """!
        @brief QPSK Modulation
        @param data_bits Input data bits
        @return Modulated QPSK symbols as np.complex64 array
        """
        print("Aufgabe 3.2: QPSK Modulation")
        modulation_data_symbols = []

        # Implement QPSK modulation

        return np.asarray(modulation_data_symbols)

    @staticmethod
    def qam16_modulation(data_bits):
        # Ensure the number of bits is even
        if len(data_bits) % 4 != 0:
            data_bits = np.append(data_bits, [0, 0, 0, 0])

        # Reshape the bits into pairs
        bit_pairs = data_bits.reshape(-1, 4)

        norm_factor = np.sqrt(10)

        # Map bit pairs to QAM16 symbols
        modulation_data_symbols = np.zeros(len(bit_pairs), dtype=complex)
        modulation_data_symbols[np.all(bit_pairs == [0, 0, 0, 0], axis=1)] = (-3 - 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 0, 0, 1], axis=1)] = (-3 - 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 0, 1, 0], axis=1)] = (-3 + 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 0, 1, 1], axis=1)] = (-3 + 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 1, 0, 0], axis=1)] = (-1 - 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 1, 0, 1], axis=1)] = (-1 - 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 1, 1, 0], axis=1)] = (-1 + 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [0, 1, 1, 1], axis=1)] = (-1 + 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 0, 0, 0], axis=1)] = (3 - 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 0, 0, 1], axis=1)] = (3 - 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 0, 1, 0], axis=1)] = (3 + 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 0, 1, 1], axis=1)] = (3 + 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 1, 0, 0], axis=1)] = (1 - 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 1, 0, 1], axis=1)] = (1 - 1j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 1, 1, 0], axis=1)] = (1 + 3j) / norm_factor
        modulation_data_symbols[np.all(bit_pairs == [1, 1, 1, 1], axis=1)] = (1 + 1j) / norm_factor



        return modulation_data_symbols

    @staticmethod
    def get_symbol_alphabet(modulation_scheme: ModulationCodingScheme):
        if modulation_scheme == ModulationCodingScheme.QPSK:
            return {
                "00": (1 + 1j) / np.sqrt(2),
                "01": (-1 + 1j) / np.sqrt(2),
                "11": (-1 - 1j) / np.sqrt(2),
                "10": (1 - 1j) / np.sqrt(2)
            }
        elif modulation_scheme == ModulationCodingScheme.BPSK:
            return {
                "0": -1,
                "1": 1
            }
        elif modulation_scheme == ModulationCodingScheme.QAM16:
            return {
                "0000": -3 - 3j,
                "0001": -3 - 1j,
                "0010": -3 + 3j,
                "0011": -3 + 1j,
                "0100": -1 - 3j,
                "0101": -1 - 1j,
                "0110": -1 + 3j,
                "0111": -1 + 1j,
                "1000": 3 - 3j,
                "1001": 3 - 1j,
                "1010": 3 + 3j,
                "1011": 3 + 1j,
                "1100": 1 - 3j,
                "1101": 1 - 1j,
                "1110": 1 + 3j,
                "1111": 1 + 1j
            }
        else:
            raise ValueError("Unknown modulation scheme")

    @staticmethod
    def plot_symbol_alphabet(modulation_scheme: ModulationCodingScheme):
        symbol_alphabet = Modulator.get_symbol_alphabet(modulation_scheme)
        plt.figure(figsize=(5, 5))

        # Scatter plot of the symbol alphabet
        plt.scatter([symbol.real for symbol in symbol_alphabet.values()],
                    [symbol.imag for symbol in symbol_alphabet.values()])

        # Annotate each symbol with its corresponding bit representation
        for bits, symbol in symbol_alphabet.items():
            plt.annotate(bits, (symbol.real, symbol.imag), textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title("Symbol Alphabet")
        plt.grid(True)
        plt.show()