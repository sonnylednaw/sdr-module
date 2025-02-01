import numpy as np
from sionna.fec.turbo import TurboEncoder, TurboDecoder


class TurboCoder:
    def __init__(self, code_rate=1/2, constraint_length=4):
        self.encoder = TurboEncoder(constraint_length=constraint_length,  # Desired constraint length of the polynomials
                               rate=code_rate,  # Desired rate of Turbo code
                               terminate=True)  # Terminate the constituent convolutional encoders to all-zero state

        # the decoder can be initialized with a reference to the encoder
        self.decoder = TurboDecoder(self.encoder,
                               num_iter=6,  # Number of iterations between component BCJR decoders
                               algorithm="map",  # can be also "maxlog"
                               hard_out=True)  # hard_decide output

    def encode(self, u):
        # if shape is not 1,n
        if len(u.shape) == 1:
            u = [np.array(u, dtype=np.int32)]
        if u.dtype != np.float32:
            received_bits = np.array(u, dtype=np.float32)
        return np.array(self.encoder(u).numpy()[0], dtype=np.int32)

    def decode(self, received_bits):
        # if shape is not 1,n
        if len(received_bits.shape) == 1:
            received_bits = [np.array(received_bits, dtype=np.float32)]
        # if not float32
        if received_bits.dtype != np.float32:
            received_bits = np.array(received_bits, dtype=np.float32)
        return np.array(self.decoder([np.array(received_bits, dtype=np.float32)]).numpy()[0], dtype=np.int32)

