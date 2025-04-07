from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator

from utilities.enums.BasebandPulseForm import BasebandPulseForm
from utilities.enums.ModulationCodingScheme import ModulationCodingScheme
from utilities.enums.SynchronizationSequences import SynchronizationSequence


class BasebandParams(BaseModel):
    T_s: float = Field(default=1 / 2, description="Dauer des Basisband Pulses in Sekunden")
    sps: int = Field(default=16, description="Anzahl der Samples pro Symbol")
    num_samps: int = Field(default=101, description="Gesamtlänge des Basisbandpulses in Samples")
    pulseform: BasebandPulseForm = Field(default=BasebandPulseForm.RAISED_COSINE,
                                         description="Form des Basisbandpulses")
    amplitude: float = Field(default=1, description="Amplitude des Pulses")
    roll_off: Optional[float] = Field(default=0.35, description="Roll-Off Faktor für Raised Cosine")
    T_sample: float = Field(default=None, description="Dauer eines Samples")

    @field_validator('num_samps')
    def check_num_samps(cls, v: int, info: ValidationInfo) -> int:
        sps = info.data.get('sps')
        if sps is not None and v <= sps:
            raise ValueError("num_samps must be greater than sps")
        return v

    @model_validator(mode='after')
    def calculate_T_sample(self) -> 'BasebandParams':
        self.T_sample = self.T_s / self.sps
        return self


class FrameParams(BaseModel):
    num_sync_syms: int = Field(default=32, description="Anzahl der Synchronisationssymbole")
    num_data_syms: int = Field(default=128, description="Anzahl der Datensymbole")
    mcs: ModulationCodingScheme = Field(default=ModulationCodingScheme.QPSK, description="Modulation Coding Scheme")
    sync_sec: SynchronizationSequence = Field(default=SynchronizationSequence.ZADOFF_CHU, description="Synchronisationssequenz")
    pilot_start_idx: int = Field(default=0, description="Startindex für Pilotsymbole")
    pilot_repetition: int = Field(default=10, description="Wiederholungsrate der Pilotsymbole")
    pilot_zc_root: int = Field(default=25, description="Wurzel des Zadoff-Chu-Pilotsequenz")


class Parameters(BaseModel):
    baseband: BasebandParams
    frame: FrameParams
