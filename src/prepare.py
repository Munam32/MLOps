"""Download Fashion-MNIST and save raw arrays to data/raw/."""

from pathlib import Path

import numpy as np
from tensorflow import keras

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.fashion_mnist.load_data()
    )

    np.savez(
        RAW_DIR / "train.npz", images=train_images, labels=train_labels
    )
    np.savez(
        RAW_DIR / "test.npz", images=test_images, labels=test_labels
    )

    print(f"Saved raw train ({train_images.shape[0]} samples) "
          f"and test ({test_images.shape[0]} samples) data to {RAW_DIR}")


if __name__ == "__main__":
    main()
