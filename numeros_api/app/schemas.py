from pydantic import BaseModel
from typing import List

class PredictResponse(BaseModel):
    digito: int
    confianza: float
    probabilidades: List[float]