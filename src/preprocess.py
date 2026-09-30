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

    train_images = train_raw["images"].astype("float32")
    train_labels = train_raw["labels"]
    test_images = test_raw["images"].astype("float32")
    test_labels = test_raw["labels"]

    if params["normalization"] == "minus_one_one":
        # Scale pixels from [0, 255] to [-1, 1] so inputs are centred on zero
        train_images = train_images / 127.5 - 1.0
        test_images = test_images / 127.5 - 1.0
        assert -1.0 <= train_images.min() and train_images.max() <= 1.0, "pixels not in [-1, 1]"
    elif params["normalization"] == "zscore":
        # Standardize with the training set's mean and std
        mean, std = train_images.mean(), train_images.std()
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std
        assert abs(train_images.mean()) < 1e-3, "training pixels not centred on 0"
    else:
        raise ValueError(f"unknown normalization: {params['normalization']}")

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
