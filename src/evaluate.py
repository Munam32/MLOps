"""Evaluate the trained model on the test set, writing metrics.json and a confusion matrix plot."""

import json
from pathlib import Path

import keras
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main():
    model = keras.models.load_model(MODELS_DIR / "model.h5")
    test_data = np.load(PROCESSED_DIR / "test.npz")
    images, labels = test_data["images"], test_data["labels"]

    test_loss, test_accuracy = model.evaluate(images, labels, verbose=0)

    predictions = model.predict(images, verbose=0).argmax(axis=1)
    matrix = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Fashion-MNIST test set (accuracy {test_accuracy:.2%})")
    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    with open(ROOT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
    print(f"Saved metrics.json to {ROOT_DIR} and confusion_matrix.png to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
