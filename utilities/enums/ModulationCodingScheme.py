from enum import Enum


class ModulationCodingScheme(Enum):
    BPSK = 1
    QPSK = 2
    QAM16 = 4
    QAM64 = 6
    QAM256 = 8

    def __str__(self):
        return self.name