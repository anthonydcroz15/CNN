import tensorflow as tf


def evaluate_model():
    # Cargar dataset
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test / 255.0

    # Cargar modelo entrenado
    model = tf.keras.models.load_model("results/model.h5")

    # Evaluar
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Guardar métricas
    with open("results/metrics.txt", "w") as f:
        f.write(f"Loss: {loss}\n")
        f.write(f"Accuracy: {accuracy}\n")

    print(f"Accuracy: {accuracy:.4f}")

