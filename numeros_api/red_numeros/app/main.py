# -*- coding: utf-8 -*-
"""
numeros_api — FastAPI
Red Neuronal: Reconocimiento de Dígitos MNIST
Alumna: Marilu Mendoza Ramírez
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictResponse
from app.model import predict_digit

app = FastAPI(
    title="Red Neuronal API — Dígitos MNIST",
    description="Reconoce dígitos escritos a mano. Alumna: Marilu Mendoza Ramírez.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Info"])
def root():
    return {
        "proyecto": "Red Neuronal — Dígitos MNIST",
        "alumna":   "Marilu Mendoza Ramírez",
        "docs":     "/docs",
        "predict":  "POST /predict",
    }

@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result   = predict_digit(contents)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")
    return PredictResponse(**result)