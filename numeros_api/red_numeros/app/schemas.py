from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    digito: int = Field(
        ge=0,
        le=9,
        description="Dígito reconocido.",
    )

    confianza: float = Field(
        ge=0,
        le=100,
        description="Confianza expresada en porcentaje.",
    )

    probabilidades: list[float] = Field(
        min_length=10,
        max_length=10,
        description=(
            "Probabilidad de cada dígito, "
            "en orden del 0 al 9."
        ),
    )