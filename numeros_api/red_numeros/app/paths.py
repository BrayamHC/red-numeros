from pathlib import Path


# Ruta absoluta de la carpeta app/
APP_DIR = Path(__file__).resolve().parent

# Modelo entrenado utilizado por FastAPI
MODEL_PATH = APP_DIR / "digits_cnn.keras"