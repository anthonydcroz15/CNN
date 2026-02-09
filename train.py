import tensorflow as tf
from model import build_lenet
import os


def train_model(epochs=10, batch_size=64):
    # Cargar dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalización
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Crear modelo
    model = build_lenet()

    # Entrenamiento
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test)
    )

    # Crear carpeta results si no existe
    if not os.path.isdir("results"):
        os.makedirs("results")

    # Guardar modelo
    model.save("results/model.h5")

    return history

