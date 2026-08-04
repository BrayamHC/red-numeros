# -*- coding: utf-8 -*-
"""
Entrena red neuronal densa para MNIST — Alumna: Marilu Mendoza Ramírez
Arquitectura: Flatten → Dense(50, relu) → Dense(50, relu) → Dense(10, softmax)
"""
import os
import math
import tensorflow as tf
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'digits.keras')

def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizar 0-255 → 0-1 y añadir canal
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    TAMANO_LOTE = 32

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(60000).batch(TAMANO_LOTE).cache().repeat()

    ds_test  = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test  = ds_test.batch(TAMANO_LOTE).cache()

    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    steps = math.ceil(60000 / TAMANO_LOTE)
    modelo.fit(ds_train, epochs=10, steps_per_epoch=steps, verbose=1)

    loss, acc = modelo.evaluate(ds_test, verbose=0)
    print(f"Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    modelo.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == '__main__':
    train()