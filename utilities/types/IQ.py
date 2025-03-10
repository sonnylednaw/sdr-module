import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel, Field
from typing import Annotated

# Definieren Sie einen benutzerdefinierten Typ für Ihre NumPy-Arrays
NumpyArrayFloat64 = Annotated[NDArray, np.float64]

class IQ(BaseModel):
    I: NumpyArrayFloat64 = Field(description="In Phase Komponente (Real)")
    Q: NumpyArrayFloat64 = Field(description="Quadratur Komponente (Imag)")
