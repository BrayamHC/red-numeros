# -*- coding: utf-8 -*-
"""
API FastAPI para reconocimiento de dígitos MNIST.

Alumna:
    Marilu Mendoza Ramírez
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_digit
from app.schemas import PredictResponse


app = FastAPI(
    title="Reconocimiento de Dígitos MNIST",
    description=(
        "API para reconocer números escritos a mano "
        "mediante una CNN entrenada con TensorFlow."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",
        "http://127.0.0.1:4321",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["Información"])
def root():
    return {
        "proyecto": "Reconocimiento de Dígitos MNIST",
        "alumna": "Marilu Mendoza Ramírez",
        "modelo": "CNN",
        "estado": "activo",
        "documentacion": "/docs",
        "prediccion": "POST /predict",
    }


@app.get("/health", tags=["Información"])
def health():
    return {
        "status": "ok",
        "service": "numeros-api",
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predicción"],
)
async def predict(file: UploadFile = File(...)):
    supported_types = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    }

    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=415,
            detail=(
                "Formato no soportado. "
                "Usa PNG, JPEG o WEBP."
            ),
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="La imagen recibida está vacía.",
            )

        result = predict_digit(image_bytes)

        return PredictResponse(**result)

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {error}",
        )