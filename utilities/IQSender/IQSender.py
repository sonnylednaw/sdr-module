import numpy as np
from numpy._typing import NDArray

from utilities.IQSender.Modulator import Modulator
from utilities.Signals import Signals
from utilities.baseband_pulses.BasebandPulse import BasebandPulse
from utilities.baseband_pulses.RaisedCosinePulse import RaisedCosinePulse
from utilities.baseband_pulses.RectPulse import RectPulse
from utilities.enums.BasebandPulseForm import BasebandPulseForm
from utilities.enums.ModulationCodingScheme import ModulationCodingScheme
from utilities.enums.ModulationCodingSchemeShift import ModulationCodingSchemeShift
from utilities.enums.SynchronizationSequences import SynchronizationSequence
from utilities.synchronization.SynchronizationSequences import SynchronizationSequences
from utilities.types.IQ import IQ
from utilities.types.Params import FrameParams, BasebandParams


class IQSender:
    """!
    @brief Class for sending IQ signals.
    """
    def __init__(self, bits: np.ndarray[np.integer], frame_params: FrameParams, baseband_params: BasebandParams):
        """!
        @brief Constructor for IQSender.
        @param bits Array of bits to be transmitted.
        @param frame_params Frame parameters.
        @param baseband_params Baseband parameters.
        """
        self.baseband_params = baseband_params
        self.frame_params = frame_params

        self.bits = bits
        self.h: np.array = self.__create_baseband_pulse(baseband_params)

        self.modulation_symbols: NDArray[np.complex64] = self.__modulate()
        self.modulation_symbols_with_pilots, self.zero_pad_indexes = self.__fill_frame()
        self.sync_seq: NDArray[np.complex64] = self.__create_sync_seq(frame_params.sync_sec, frame_params.num_sync_syms)
        self.frame_modulation_symbols: NDArray[np.complex64] = self.__concatenate_sync_and_data_symbols()

        self.x_iq_no_shape: IQ = self.__dirac_sum_with_frame_symbols(self.frame_modulation_symbols)

        self.x_iq_shaped: IQ = self.__shape_symbols(self.x_iq_no_shape, self.h)


    def __create_baseband_pulse(self, baseband_params: BasebandParams) -> BasebandPulse:
        """!
        @brief Create baseband pulse.
        @param baseband_params Baseband parameters.
        @return Baseband pulse.
        """
        pulse_generators = {
            BasebandPulseForm.RECT: RectPulse,
            BasebandPulseForm.RAISED_COSINE: RaisedCosinePulse
        }

        pulse_class = pulse_generators.get(self.baseband_params.pulseform)
        if pulse_class is None:
            raise ValueError("Unknown baseband pulse form")

        pulse = pulse_class.from_baseband_params(baseband_params=baseband_params)

        self.h = pulse.generate_pulse()
        return pulse

    def __modulate(self):
        """!
        @brief Modulates the input bits.
        @return Modulated symbols.
        """
        modulation_methods = {
            ModulationCodingScheme.QPSK: Modulator.qpsk_modulation,
            ModulationCodingScheme.BPSK: Modulator.bpsk_modulation,
            ModulationCodingScheme.QAM16: Modulator.qam16_modulation
        }

        modulation_method = modulation_methods.get(self.frame_params.mcs)
        if modulation_method is None:
            raise ValueError("Unknown modulation scheme")

        self.modulation_symbols = modulation_method(self.bits)
        return self.modulation_symbols

    def __fill_frame(self):
        """!
        @brief Fills the frame with modulation symbols and pilots.
        @return Tuple of modulation symbols with pilots and zero padding indexes.
        """
        # TODO: Als Aufgabe erstellen
        print("Aufgabe: Multiplexing von Piloten. Implementieren Sie die Funktion.")
        total_symbols = self.frame_params.num_data_syms
        pilot_start_idx = self.frame_params.pilot_start_idx
        pilot_repetition = self.frame_params.pilot_repetition

        # Berechnung der Anzahl der Piloten

        modulation_data_symbols_with_pilots = np.zeros(self.frame_params.num_data_syms, dtype=np.complex64)

        pilot_idx = np.arange(pilot_start_idx, total_symbols, pilot_repetition + 1)
        pilot_seq = self.__create_pilot_seq(pilot_idx.size)

        data_sym_idx = 0
        j = 0

        zero_padding_indexes = []

        for i in range(len(modulation_data_symbols_with_pilots)):
            if np.isin(i, pilot_idx):
                modulation_data_symbols_with_pilots[i] = pilot_seq[j]
                j += 1
            elif data_sym_idx < self.modulation_symbols.size:
                modulation_data_symbols_with_pilots[i] = self.modulation_symbols[data_sym_idx]
                data_sym_idx += 1
            else:
                # zero padding
                modulation_data_symbols_with_pilots[i] = 0
                zero_padding_indexes.append(i)
        self.modulation_symbols_with_pilots = modulation_data_symbols_with_pilots
        return self.modulation_symbols_with_pilots, np.asarray(zero_padding_indexes)

    def __create_pilot_seq(self, length) -> NDArray[np.complex64]:
        """!
        @brief Creates the pilot sequence.
        @param length Length of the pilot sequence.
        @return Generated pilot sequence.
        """
        mcs_shift = {
            ModulationCodingScheme.BPSK: ModulationCodingSchemeShift.BPSK,
            ModulationCodingScheme.QPSK: ModulationCodingSchemeShift.QPSK,
            ModulationCodingScheme.QAM16: ModulationCodingSchemeShift.QAM16
        }.get(self.frame_params.mcs)

        if mcs_shift is None:
            raise ValueError("Unknown modulation coding scheme")

        pilot_zc_seq = SynchronizationSequences.zadoff_chu_sequence(length=length, root=self.frame_params.pilot_zc_root)
        return np.roll(pilot_zc_seq, mcs_shift.value)

    def __create_sync_seq(self, sync_seq_type: SynchronizationSequence, length: int = 16) -> NDArray[np.complex64]:
        """!
        @brief Creates the synchronization sequence.
        @param sync_seq_type Type of synchronization sequence.
        @param length Length of the synchronization sequence.
        @return Generated synchronization sequence.
        """
        sync_sequences = {
            SynchronizationSequence.ZADOFF_CHU: SynchronizationSequences.zadoff_chu_sequence(length),
            SynchronizationSequence.M_SEQUENCE: SynchronizationSequences.generate_m_sequence(length)
        }

        sync_sequence = sync_sequences.get(sync_seq_type)

        if sync_sequence is None:
            raise ValueError("Unknown synchronization sequence")

        self.sync_seq = sync_sequence
        return self.sync_seq

    def __concatenate_sync_and_data_symbols(self) -> NDArray[np.complex64]:
        """!
        @brief Concatenates synchronization and data symbols.
        @return Concatenated frame modulation symbols.
        """
        print("Aufgabe: Bilden Sie die Modulationssymbole mit den Synchronisationssymbolen.")
        self.frame_modulation_symbols = np.concatenate((self.sync_seq, self.modulation_symbols_with_pilots))
        return self.frame_modulation_symbols

    def __dirac_sum_with_frame_symbols(self, modulation_symbols: NDArray[np.complex64]) -> IQ:
        """!
        @brief Forms the Dirac impulse sequence with modulation symbols.
        @param modulation_symbols Modulation symbols.
        @return IQ samples without shaping.
        """
        # TODO: Aufgabe erstellen diese Methode zu implementieren
        print("Aufgabe: Bilden Sie die Dirac-Impulsfolge mit den Modulationssymbolen.")

        x_no_shaped = IQ(
            I=np.zeros(self.frame_modulation_symbols.size * self.baseband_params.sps),
            Q=np.zeros(self.frame_modulation_symbols.size * self.baseband_params.sps)
        )

        t = np.arange(x_no_shaped.I.size) * self.baseband_params.T_sample
        for m in range(modulation_symbols.size):
            x_no_shaped.I += modulation_symbols[m].real * Signals.delta_distribution(t=t,
                                                                                     tau=m * self.baseband_params.T_s)
            x_no_shaped.Q += modulation_symbols[m].imag * Signals.delta_distribution(t=t,
                                                                                     tau=m * self.baseband_params.T_s)

        self.x_iq_no_shape = x_no_shaped

        return self.x_iq_no_shape

    def __shape_symbols(self, x_iq_no_shape: IQ, h: BasebandPulse) -> IQ:
        """!
        @brief Shapes the symbols with the baseband pulse.
        @param x_iq_no_shape IQ samples without shaping.
        @param h Baseband pulse.
        @return Shaped IQ samples.
        """
        # TODO: Aufgabe erstellen diese Methode zu implementieren
        print("Aufgabe: Formen Sie die Symbole mit dem Basisbandpuls.")

        x_iq_shaped = IQ(
            I=np.convolve(x_iq_no_shape.I, h.generate_pulse()),
            Q=np.convolve(x_iq_no_shape.Q, h.generate_pulse())
        )

        self.x_iq_shaped = x_iq_shaped
        return self.x_iq_shaped

    def get_samples_as_complex64(self) -> NDArray[np.complex64]:
        """!
        @brief Returns the IQ samples as complex64.
        @return IQ samples as complex64.
        """
        return self.x_iq_shaped.I + 1j * self.x_iq_shaped.Q
