from enum import Enum


class SynchronizationSequence(Enum):
    ZADOFF_CHU = 1
    GOLD = 2
    BARKER = 4
    M_SEQUENCE = 6

    def __str__(self):
        return self.name