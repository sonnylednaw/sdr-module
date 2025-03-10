import numpy as np


class SynchronizationSequences:
    DEFAULT_LENGTH = 16

    @staticmethod
    def generate_m_sequence(length=DEFAULT_LENGTH):
        """
        Generates a 16-bit long M-sequence with LFSR (Linear Feedback Shift Register) as BPSK sequence.
        Used primitive polynomial: x^4 + x^3 + 1 (Taps at positions 3 and 4)
        """
        register = [1, 1, 1, 1]  # Initial state (must not be all 0)
        sequence = []

        for _ in range(length):
            # Feedback calculation (XOR of the taps)
            feedback = register[3] ^ register[2]  # 0-indexed positions
            sequence.append(register.pop())
            register.insert(0, feedback)

        return 2 * np.asarray(sequence) - 1

    @staticmethod
    def zadoff_chu_sequence(length=DEFAULT_LENGTH, root=25):
        """
        Generates a ZC-sequence (QPSK) for a given length and root.
        :param length:
        :param root:
        :return:
        """
        n = np.arange(0, length)
        if length % 2 != 0:
            return np.exp(1j * np.pi * root * n * (n + 1) / length)
        return np.exp(1j * np.pi * root * n ** 2 / length)
