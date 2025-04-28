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
        self.h: np.array = self.create_baseband_pulse(baseband_params)

        self.modulation_symbols: NDArray[np.complex64] = self.modulate()
        self.modulation_symbols_with_pilots, self.zero_pad_indexes = self.fill_frame()
        self.sync_seq: NDArray[np.complex64] = self.create_sync_seq()
        self.frame_modulation_symbols: NDArray[np.complex64] = self.concatenate_sync_and_data_symbols()

        self.x_iq_no_shape: IQ = self.dirac_sum_with_frame_symbols()

        self.x_iq_shaped: IQ = self.shape_symbols()


    def create_baseband_pulse(self, baseband_params: BasebandParams) -> BasebandPulse:
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

    def modulate(self):
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

    def fill_frame(self):
        """!
        @brief Fills the frame with modulation symbols and pilots.
        @return Tuple of modulation symbols with pilots and zero padding indexes.
        """
        print("Aufgabe 4.1: Befüllung des Frames.")

        self.modulation_symbols_with_pilots = np.zeros(self.frame_params.num_data_syms, dtype=np.complex64)
        # pilot_seq = self.create_pilot_seq(number_of_pilots)
        zero_padding_indexes = []

        # Fill the frame with modulation symbols and pilots

        return self.modulation_symbols_with_pilots, np.asarray(zero_padding_indexes)

    def create_pilot_seq(self, length) -> NDArray[np.complex64]:
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

    def create_sync_seq(self) -> NDArray[np.complex64]:
        """!
        @brief Creates the synchronization sequence.
        @param sync_seq_type Type of synchronization sequence.
        @param length Length of the synchronization sequence.
        @return Generated synchronization sequence.
        """

        sync_sequences = {
            SynchronizationSequence.ZADOFF_CHU: SynchronizationSequences.zadoff_chu_sequence(self.frame_params.num_sync_syms),
            SynchronizationSequence.M_SEQUENCE: SynchronizationSequences.generate_m_sequence(self.frame_params.num_sync_syms)
        }

        sync_sequence = sync_sequences.get(self.frame_params.sync_sec)

        if sync_sequence is None:
            raise ValueError("Unknown synchronization sequence")

        self.sync_seq = sync_sequence
        return self.sync_seq

    def concatenate_sync_and_data_symbols(self) -> NDArray[np.complex64]:
        """!
        @brief Concatenates synchronization and data symbols.
        @return Concatenated frame modulation symbols.
        """
        print("Aufgabe 6: Bilden Sie die Modulationssymbole mit den Synchronisationssymbolen.")

        frame_modulation_symbols = []

        # concatenate self.sync_seq and self.modulation_symbols_with_pilots in the return array

        return np.asarray(frame_modulation_symbols, dtype=np.complex64)

    def dirac_sum_with_frame_symbols(self) -> IQ:
        """!
        @brief Forms the Dirac impulse sequence with modulation symbols.
        @param modulation_symbols Modulation symbols.
        @return IQ samples without shaping.
        """

        print("Aufgabe 7: Bilden Sie die Dirac-Impulsfolge mit den Modulationssymbolen.")

        modulation_symbols = self.frame_modulation_symbols

        x_no_shaped = IQ(
            I=np.zeros(self.frame_modulation_symbols.size * self.baseband_params.sps),
            Q=np.zeros(self.frame_modulation_symbols.size * self.baseband_params.sps)
        )

        print("Aufgabe 7: Bilden Sie die Dirac-Impulsfolge mit den Modulationssymbolen.")

        # Implement the Dirac impulse sequence with modulation symbols here

        self.x_iq_no_shape = x_no_shaped

        return self.x_iq_no_shape

    def shape_symbols(self) -> IQ:
        """!
        @brief Shapes the symbols with the baseband pulse.
        @param x_iq_no_shape IQ samples without shaping.
        @param h Baseband pulse.
        @return Shaped IQ samples.
        """

        h = self.h.generate_pulse()
        x_iq_shaped: IQ = IQ(
            I=np.zeros(self.x_iq_no_shape.I.size),
            Q=np.zeros(self.x_iq_no_shape.Q.size)
        )

        print("Aufgabe 8: Formen Sie die Symbole mit dem Basisbandpuls.")

        #x_iq_shaped = IQ(
        #    I= # Convolve the I component with the pulse shape as NumPy array
        #    Q= # Convolve the Q component with the pulse shape as NumPy arrayﬁ
        #)

        self.x_iq_shaped = x_iq_shaped
        return self.x_iq_shaped

    def get_samples_as_complex64(self) -> NDArray[np.complex64]:
        """!
        @brief Returns the IQ samples as complex64.
        @return IQ samples as complex64.
        """
        return self.x_iq_shaped.I + 1j * self.x_iq_shaped.Q
