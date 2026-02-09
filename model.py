
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lenet(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential(name="LeNet")

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(6, (5, 5), activation="relu"))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(16, (5, 5), activation="relu"))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="relu"))
    model.add(layers.Dense(84, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
