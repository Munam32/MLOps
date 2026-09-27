"""Normalize raw Fashion-MNIST arrays and split a validation set."""

from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def main():
    with open(ROOT_DIR / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["preprocess"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = np.load(RAW_DIR / "train.npz")
    test_raw = np.load(RAW_DIR / "test.npz")

    train_images = train_raw["images"].astype("float32") / 255.0
    train_labels = train_raw["labels"]
    test_images = test_raw["images"].astype("float32") / 255.0
    test_labels = test_raw["labels"]

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=train_labels,
    )

    np.savez(PROCESSED_DIR / "train.npz", images=train_images, labels=train_labels)
    np.savez(PROCESSED_DIR / "val.npz", images=val_images, labels=val_labels)
    np.savez(PROCESSED_DIR / "test.npz", images=test_images, labels=test_labels)

    print(
        f"Saved processed train ({train_images.shape[0]}), "
        f"val ({val_images.shape[0]}), and test ({test_images.shape[0]}) "
        f"data to {PROCESSED_DIR}"
    )


if __name__ == "__main__":
    main()
