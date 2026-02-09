print("🚀 MAIN INICIADO")

from train import train_model
from evaluate import evaluate_model

print("📦 Imports OK")

if __name__ == "__main__":
    print("🔥 Entrenando modelo...")
    train_model(epochs=1)

    print("🧪 Evaluando modelo...")
    evaluate_model()

    print("✅ Proceso finalizado")

