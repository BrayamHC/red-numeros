import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'digits.keras')
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _centrar_como_mnist(arr: np.ndarray) -> np.ndarray:
    """
    Recorta el bounding box del dígito, lo reescala a 20x20
    y lo centra en un lienzo de 28x28 — igual que hace MNIST
    con sus imágenes originales.
    """
    filas = np.any(arr > 0.05, axis=1)
    cols  = np.any(arr > 0.05, axis=0)

    if not filas.any() or not cols.any():
        return arr  # canvas vacío, se devuelve tal cual

    y0, y1 = np.where(filas)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    recorte = arr[y0:y1 + 1, x0:x1 + 1]

    img_recorte = Image.fromarray((recorte * 255).astype('uint8'))
    lado_max = max(img_recorte.size)
    escala = 20.0 / lado_max
    nuevo_w = max(1, int(img_recorte.width * escala))
    nuevo_h = max(1, int(img_recorte.height * escala))
    img_recorte = img_recorte.resize((nuevo_w, nuevo_h))

    lienzo = Image.new('L', (28, 28), 0)
    offset_x = (28 - nuevo_w) // 2
    offset_y = (28 - nuevo_h) // 2
    lienzo.paste(img_recorte, (offset_x, offset_y))

    return np.array(lienzo).astype('float32') / 255.0


def predict_digit(image_bytes: bytes) -> dict:
    model = load_model()

    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    arr = np.array(img).astype('float32') / 255.0

    # NO se vuelve a invertir: el canvas ya envía trazo claro / fondo oscuro,
    # igual que MNIST. Antes aquí había un `1.0 - arr` que invertía dos veces.
    arr = _centrar_como_mnist(arr)
    arr = arr.reshape(1, 28, 28, 1)

    probs  = model.predict(arr, verbose=0)[0]
    digito = int(np.argmax(probs))

    return {
        "digito":         digito,
        "confianza":      round(float(probs[digito]) * 100, 2),
        "probabilidades": [round(float(p) * 100, 2) for p in probs],
    }