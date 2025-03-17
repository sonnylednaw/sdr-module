from enum import Enum
from utilities.enums.ModulationCodingScheme import ModulationCodingScheme

class ModulationCodingSchemeShift(Enum):
    BPSK = 4
    QPSK = 6
    QAM16 = 8