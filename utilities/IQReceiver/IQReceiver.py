from typing import Any

import numpy as np
from numpy import ndarray, dtype

from utilities.IQReceiver.Demodulator import Demodulator
from utilities.OSUtilities import print_clickable_link
from utilities.enums.ModulationCodingScheme import ModulationCodingScheme
from utilities.enums.ModulationCodingSchemeShift import ModulationCodingSchemeShift
from utilities.synchronization.SynchronizationSequences import SynchronizationSequences
from utilities.types.Params import Parameters


class IQReceiver:

    def __init__(self, params: Parameters, symbols_without_zero_padding: int, sdr_samples: np.ndarray[np.complex64] = None, known_sync_seq: np.ndarray[np.complex64] = None):
        self.sdr_samples = sdr_samples
        self.params = params
        self.known_sync_seq = known_sync_seq
        self.symbols_without_zero_padding = symbols_without_zero_padding
        
        self.tau0, self.corr = self.time_synchronization()
        self.frame_samples = self.extract_frame_samples()


        self.base_pilot_seq = self.get_base_pilot_seq()
        self.pilot_seq = self.demultiplex_pilots()

        self.cyclic_correlation, self.mcs_shift = self.estimate_mcs_shift()
        self.used_pilot_seq = self.calculate_used_pilot_seq()
        
        self.frame_samples_eq = self.channel_equalization()

        self.frame_samples_without_pilots = self.remove_pilot_symbols()

        self.frame_samples_without_zero_padding = self.remove_zero_padded_symbols()

        self.demodulated_bits = self.demodulate()

    def set_samples(self, sdr_samples: np.ndarray[np.complex64]):
        self.sdr_samples = sdr_samples

    def time_synchronization(self):
        """!
        @brief Time synchronization of the received samples.
        Input object variables: self.sdr_samples, self.known_sync_seq
        Output: tau0 (Peak of correlation), corr (Correlation values)
        """

        print("Aufgabe 19: Korrelationsempfänger")

        self.corr = np.zeros(len(self.sdr_samples))

        tau0 = 0

        # Implement the correlation receiver

        return tau0, self.corr

    def extract_frame_samples(self):
        """!
        @brief Extract the frame samples from the received samples.
        Input: self.sdr_samples, self.tau0, self.params
        Output: Frame samples
        """
        print("Aufgabe 20: Extraktion der Rahmendaten")

        frame_samples = []

        return frame_samples # should be a numpy array of complex64

    def demultiplex_pilots(self):
        """!
        @brief Demultiplex the pilot symbols from the received samples.
        Input: self.frame_samples, self.params
        Output: Pilot symbols
        """
        print("Aufgabe 22: Extraktion der Pilotensequenz")

        pilot_seq = []

        # Implement the demultiplexing of pilot symbols from the frame samples

        return np.array(pilot_seq)

    def get_base_pilot_seq(self):
        """!
        @brief Get the base pilot sequence.
        Input : self.params
        Output: Base pilot sequence
        """
        print("Aufgabe 21: Generierung der Basis Piloten-Sequenz")
        pilot_seq = []

        return pilot_seq # should be a numpy array of complex64

    def estimate_mcs_shift(self) -> tuple[np.ndarray[Any, np.dtype[np.complexfloating]], np.signedinteger]:
        """!
        @brief Estimate the modulation coding scheme shift.
        Input: self.base_pilot_seq, self.pilot_seq
        Output: Cyclic correlation, MCS shift
        """

        print("Aufgabe 23: Zyklische Korrelation")

        cyclic_correlation_normed = np.zeros(self.base_pilot_seq.size, dtype=np.complex64)

        # Implement the cyclic correlation calculation
        # Norm the cyclic correlation to the maximum value

        return cyclic_correlation_normed, np.argmax(cyclic_correlation_normed)

    def calculate_used_pilot_seq(self) -> ndarray[Any, dtype[Any]]:
        """!
        @brief Calculate the used pilot sequence.
        Input: self.base_pilot_seq, self.mcs_shift
        Output: Used pilot sequence
        """

        print("Aufgabe 24: Berechnung der tatsächlich verwendeten Pilotensequenz")
        used_pilot_seq = np.asarray([])

        # Implement the calculation of the used pilot sequence

        return used_pilot_seq

    def channel_equalization(self) -> ndarray[np.complex64]:
        """!
        @brief Channel equalization of the received samples.
        Input: self.frame_samples, self.used_pilot_seq, self.params
        Output: Equalized frame samples
        """
        print("Aufgabe 25: Kanalschätzung und Equalization")
        frame_samples_eq = np.zeros_like(self.frame_samples, np.complex64)

        # Implement the channel equalization
            
        return frame_samples_eq

    def remove_pilot_symbols(self):
        """!
        @brief Remove the pilot symbols from the frame samples.
        Input: self.frame_samples, self.params
        Output: Frame samples without pilot symbols
        """
        print("Aufgabe 26: Auslesen der entzerrten Datensymbole")

        modulation_data_symbols = []

        # Implement the removal of pilot symbols from the frame samples

        return np.array(modulation_data_symbols)

    def remove_zero_padded_symbols(self) -> ndarray[Any, dtype[Any]]:
        """!
        @brief Remove the zero-padded symbols from the frame samples.
        Input: self.frame_samples, self.symbols_without_zero_padding
        Output: Frame samples without zero-padded symbols
        """
        print("Aufgabe 27: Entfernen der Zero-Padding-Symbole")

        frame_samples_without_zero_padding_and_pilots = np.asarray([])

        # Implement the removal of zero-padded symbols from the frame samples

        return frame_samples_without_zero_padding_and_pilots

    def demodulate(self) -> ndarray[np.int8]:
        """!
        @brief Demodulate the received samples.
        Input: self.frame_samples_without_zero_padding, self.mcs_shift
        Output: Demodulated bits
        """
        demodulation_methods = {
            ModulationCodingScheme.BPSK: Demodulator.bpsk_decision,
            ModulationCodingScheme.QPSK: Demodulator.qpsk_decision
        }

        if self.mcs_shift == ModulationCodingSchemeShift.BPSK.value:
            demodulation_method = demodulation_methods.get(ModulationCodingScheme.BPSK)
        elif self.mcs_shift == ModulationCodingSchemeShift.QPSK.value:
            demodulation_method = demodulation_methods.get(ModulationCodingScheme.QPSK)
        else:
            raise ValueError("Unknown modulation scheme")

        return demodulation_method(self.frame_samples_without_zero_padding)
