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

        # TODO: Nachfolgenden Code entfernen

        self.corr = np.zeros(len(self.sdr_samples))

        for i in range(len(self.sdr_samples) - len(self.known_sync_seq)):
            self.corr[i] = np.dot(self.sdr_samples[i:i + len(self.known_sync_seq)],
                             np.conjugate(self.known_sync_seq))

        return self.corr.argmax(), self.corr

    def extract_frame_samples(self):
        """!
        @brief Extract the frame samples from the received samples.
        Input: self.sdr_samples, self.tau0, self.params
        Output: Frame samples
        """
        print("Aufgabe 20: Extraktion der Rahmendaten")
        # TODO: Lösung entfernen
        return self.sdr_samples[
                   self.tau0 + self.params.frame.num_sync_syms:
                   self.tau0 + self.params.frame.num_sync_syms + self.params.frame.num_data_syms
               ]

    def demultiplex_pilots(self):
        """!
        @brief Demultiplex the pilot symbols from the received samples.
        Input: self.frame_samples, self.params
        Output: Pilot symbols
        """
        print("Aufgabe 22: Extraktion der Pilotensequenz")
        # TODO: Reutrn stehen lassen aber code dazwischen entfernen

        pilot_start_idx = self.params.frame.pilot_start_idx
        total_symbols = self.params.frame.num_data_syms
        pilot_repetition = self.params.frame.pilot_repetition

        pilot_seq = []

        pilot_idx = np.arange(pilot_start_idx, total_symbols, pilot_repetition + 1)

        for i in range(len(self.frame_samples)):
            if np.isin(i, pilot_idx):
                pilot_seq.append(self.frame_samples[i])

        return np.array(pilot_seq)

    def get_base_pilot_seq(self):
        """!
        @brief Get the base pilot sequence.
        Input : self.params
        Output: Base pilot sequence
        """
        print("Aufgabe 21: Generierung der Basis Piloten-Sequenz")
        total_pilots = np.arange(self.params.frame.pilot_start_idx, self.params.frame.num_data_syms,
                                 self.params.frame.pilot_repetition + 1).size

        return SynchronizationSequences.zadoff_chu_sequence(length=total_pilots, root=self.params.frame.pilot_zc_root)

    def estimate_mcs_shift(self) -> tuple[np.ndarray[Any, np.dtype[np.complexfloating]], np.signedinteger]:
        """!
        @brief Estimate the modulation coding scheme shift.
        Input: self.base_pilot_seq, self.pilot_seq
        Output: Cyclic correlation, MCS shift
        """

        print("Aufgabe 23: Zyklische Korrelation")
        # TODO: Lösung entfernen, return stehen lassen

        cyclic_correlation = np.fft.ifft(
            np.fft.fft(self.base_pilot_seq) * np.fft.fft(np.conjugate(self.pilot_seq))
        )

        cyclic_correlation_normed = cyclic_correlation / np.max(cyclic_correlation)

        return cyclic_correlation_normed, np.argmax(cyclic_correlation_normed)

    def calculate_used_pilot_seq(self) -> ndarray[Any, dtype[Any]]:
        """!
        @brief Calculate the used pilot sequence.
        Input: self.base_pilot_seq, self.mcs_shift
        Output: Used pilot sequence
        """

        print("Aufgabe 24: Berechnung der tatsächlich verwendeten Pilotensequenz")
        used_pilot_seq = np.roll(self.base_pilot_seq, self.mcs_shift)

        return used_pilot_seq

    def channel_equalization(self) -> ndarray[np.complex64]:
        """!
        @brief Channel equalization of the received samples.
        Input: self.frame_samples, self.used_pilot_seq, self.params
        Output: Equalized frame samples
        """
        print("Aufgabe 25: Kanalschätzung und Equalization")
        # TODO: Lösung entfernen, return stehen lassen
        frame_samples_eq = np.zeros_like(self.frame_samples, np.complex64)

        pilot_indexes = np.arange(self.params.frame.pilot_start_idx, self.params.frame.num_data_syms, self.params.frame.pilot_repetition + 1)

        for pilot_idx, j in zip(pilot_indexes, range(self.used_pilot_seq.size)):
            curr_rx_pilot = self.frame_samples[pilot_idx]
            known_pilot = self.used_pilot_seq[j]

            curr_h = (curr_rx_pilot * np.conjugate(known_pilot)) / (np.abs(known_pilot)**2)
            data_sym_slice = slice(pilot_idx + 1, pilot_idx + self.params.frame.pilot_repetition + 1)
            frame_samples_eq[data_sym_slice] = (self.frame_samples[data_sym_slice] * np.conjugate(curr_h)) / (np.abs(curr_h)**2)
            frame_samples_eq[pilot_idx] = self.used_pilot_seq[j]
            
        return frame_samples_eq

    def remove_pilot_symbols(self):
        """!
        @brief Remove the pilot symbols from the frame samples.
        Input: self.frame_samples, self.params
        Output: Frame samples without pilot symbols
        """
        print("Aufgabe 26: Auslesen der entzerrten Datensymbole")
        # TODO: Lösung entfernen, return stehen lassen

        pilot_start_idx = self.params.frame.pilot_start_idx
        pilot_repetition = self.params.frame.pilot_repetition

        modulation_data_symbols = []

        for i in range(len(self.frame_samples)):
            if (i - pilot_start_idx) % (pilot_repetition + 1) != 0:
                modulation_data_symbols.append(self.frame_samples[i])

        return np.array(modulation_data_symbols)

    def remove_zero_padded_symbols(self) -> ndarray[Any, dtype[Any]]:
        """!
        @brief Remove the zero-padded symbols from the frame samples.
        Input: self.frame_samples, self.symbols_without_zero_padding
        Output: Frame samples without zero-padded symbols
        """
        print("Aufgabe 27: Entfernen der Zero-Padding-Symbole")
        # TODO: Lösung entfernen, return stehen lassen
        return self.frame_samples_without_pilots[:self.symbols_without_zero_padding]

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
