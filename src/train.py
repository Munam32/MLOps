"""Build and train the Fashion-MNIST ANN, saving the model and training history."""

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


def main():
    with open(ROOT_DIR / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_data = np.load(PROCESSED_DIR / "train.npz")
    val_data = np.load(PROCESSED_DIR / "val.npz")

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(params["dense_units"], activation="relu"),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_data["images"],
        train_data["labels"],
        validation_data=(val_data["images"], val_data["labels"]),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model.save(MODELS_DIR / "model.h5")
    pd.DataFrame(history.history).to_csv(MODELS_DIR / "history.csv", index_label="epoch")

    print(f"Saved trained model and history to {MODELS_DIR}")


if __name__ == "__main__":
    main()
