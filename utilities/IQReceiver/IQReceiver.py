import numpy as np

from utilities.OSUtilities import print_clickable_link
from utilities.synchronization.SynchronizationSequences import SynchronizationSequences
from utilities.types.Params import Parameters


class IQReceiver:

    def __init__(self, params: Parameters, sdr_samples: np.ndarray[np.complex64] = None, known_sync_seq: np.ndarray[np.complex64] = None):
        self.sdr_samples = sdr_samples
        self.params = params
        self.known_sync_seq = known_sync_seq
        
        self.tau0, self.corr = self.time_synchronization()
        self.frame_samples = self.extract_frame_samples()


        self.base_pilot_seq = self.get_base_pilot_seq()
        self.pilot_seq = self.demultiplex_pilots()

    def set_samples(self, sdr_samples: np.ndarray[np.complex64]):
        self.sdr_samples = sdr_samples

    def time_synchronization(self):
        """!
        @brief Time synchronization of the received samples.
        """

        print_clickable_link('utilities/IQReceiver/IQReceiver.py', 10)

        self.corr = np.zeros(len(self.sdr_samples))

        for i in range(len(self.sdr_samples) - len(self.known_sync_seq)):
            self.corr[i] = np.dot(self.sdr_samples[i:i + len(self.known_sync_seq)],
                             np.conjugate(self.known_sync_seq))

        return self.corr.argmax(), self.corr

    def extract_frame_samples(self):
        """!
        @brief Extract the frame samples from the received samples.
        """
        return self.sdr_samples[
                   self.tau0 + self.params.frame.num_sync_syms:
                   self.tau0 + self.params.frame.num_sync_syms + self.params.frame.num_data_syms
               ]

    def demultiplex_pilots(self):
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
        """
        total_pilots = np.arange(self.params.frame.pilot_start_idx, self.params.frame.num_data_syms,
                                 self.params.frame.pilot_repetition + 1).size

        return SynchronizationSequences.zadoff_chu_sequence(length=total_pilots, root=self.params.frame.pilot_zc_root)