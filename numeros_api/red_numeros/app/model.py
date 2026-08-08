# -*- coding: utf-8 -*-
"""
Carga del modelo CNN y predicción de dígitos.
"""

import io
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from app.paths import MODEL_PATH


_model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    """
    Carga el modelo una sola vez y lo reutiliza.
    """
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en: {MODEL_PATH}"
            )

        _model = tf.keras.models.load_model(MODEL_PATH)

        print(f"Modelo cargado desde: {MODEL_PATH}")

    return _model


def image_to_array(image_bytes: bytes) -> np.ndarray:
    """
    Convierte una imagen recibida a un array normalizado.

    El formato esperado es:

        fondo oscuro = 0
        trazo claro = 1
    """
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    # El frontend puede enviar 280x280,
    # pero el modelo requiere 28x28.
    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS,
    )

    # Suavizado muy ligero para conservar trazos delgados
    # después del redimensionamiento.
    image = image.filter(
        ImageFilter.GaussianBlur(radius=0.20)
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    return array


def center_digit(image_array: np.ndarray) -> np.ndarray:
    """
    Recorta el dígito, lo escala proporcionalmente
    y lo centra en un lienzo de 28x28.
    """
    threshold = 0.08

    rows = np.any(image_array > threshold, axis=1)
    cols = np.any(image_array > threshold, axis=0)

    # Canvas vacío.
    if not rows.any() or not cols.any():
        return np.zeros(
            (28, 28),
            dtype=np.float32,
        )

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    cropped = image_array[
        y0:y1 + 1,
        x0:x1 + 1,
    ]

    cropped_image = Image.fromarray(
        np.clip(cropped * 255.0, 0, 255).astype("uint8"),
        mode="L",
    )

    max_side = max(cropped_image.size)
    scale = 20.0 / max_side

    new_width = max(
        1,
        round(cropped_image.width * scale),
    )

    new_height = max(
        1,
        round(cropped_image.height * scale),
    )

    cropped_image = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    centered = Image.new(
        mode="L",
        size=(28, 28),
        color=0,
    )

    offset_x = (28 - new_width) // 2
    offset_y = (28 - new_height) // 2

    centered.paste(
        cropped_image,
        (offset_x, offset_y),
    )

    return np.asarray(
        centered,
        dtype=np.float32,
    ) / 255.0


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Devuelve un tensor con forma:

        (1, 28, 28, 1)
    """
    if not image_bytes:
        raise ValueError(
            "La imagen recibida está vacía."
        )

    image_array = image_to_array(image_bytes)

    # No invertir la imagen.
    #
    # El canvas debe enviar:
    #   fondo negro + trazo blanco
    #
    # Ese formato coincide con MNIST.
    image_array = center_digit(image_array)

    return image_array.reshape(
        1,
        28,
        28,
        1,
    )


def predict_digit(image_bytes: bytes) -> dict[str, Any]:
    """
    Genera una predicción para la imagen recibida.
    """
    model = load_model()
    tensor = preprocess_image(image_bytes)

    probabilities = model.predict(
        tensor,
        verbose=0,
    )[0]

    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    return {
        "digito": digit,
        "confianza": round(
            confidence * 100,
            2,
        ),
        "probabilidades": [
            round(
                float(probability) * 100,
                2,
            )
            for probability in probabilities
        ],
    }