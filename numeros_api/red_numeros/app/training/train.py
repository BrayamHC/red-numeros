# -*- coding: utf-8 -*-
"""
Entrenamiento de una red neuronal convolucional para MNIST.

Alumna:
    Marilu Mendoza Ramírez

El modelo reconoce dígitos escritos a mano del 0 al 9.
"""

import tensorflow as tf

from app.paths import MODEL_PATH


IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 10
SEED = 42


def load_datasets():
    """
    Descarga MNIST y normaliza los píxeles al rango 0-1.
    """
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Agregar canal de escala de grises.
    # Antes: (cantidad, 28, 28)
    # Después: (cantidad, 28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(
            buffer_size=len(x_train),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, test_dataset


def build_model() -> tf.keras.Model:
    """
    Construye la CNN utilizada para clasificar los dígitos.
    """
    augmentation = tf.keras.Sequential(
        [
            # Rotaciones pequeñas para tolerar escritura inclinada.
            tf.keras.layers.RandomRotation(
                factor=0.05,
                fill_mode="constant",
                fill_value=0.0,
            ),

            # Desplazamientos pequeños para tolerar dígitos
            # que no estén perfectamente centrados.
            tf.keras.layers.RandomTranslation(
                height_factor=0.06,
                width_factor=0.06,
                fill_mode="constant",
                fill_value=0.0,
            ),

            # Variaciones pequeñas de tamaño.
            tf.keras.layers.RandomZoom(
                height_factor=(-0.06, 0.06),
                width_factor=(-0.06, 0.06),
                fill_mode="constant",
                fill_value=0.0,
            ),
        ],
        name="data_augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
            ),

            # Solo se activa durante el entrenamiento.
            augmentation,

            # Primer bloque convolucional.
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.15),

            # Segundo bloque convolucional.
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.20),

            # Tercer bloque convolucional.
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.BatchNormalization(),

            # Reduce los parámetros respecto a Flatten.
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),

            # Diez clases: 0, 1, 2, ..., 9.
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train():
    """
    Entrena el modelo y guarda la mejor versión.
    """
    tf.keras.utils.set_random_seed(SEED)

    train_dataset, test_dataset = load_datasets()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.summary()

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=0,
    )

    print(f"Accuracy de prueba: {test_accuracy:.4f}")
    print(f"Loss de prueba: {test_loss:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    train()