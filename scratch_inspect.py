"""Scratch: quick look at raw array shapes while building the pipeline."""

import numpy as np

data = np.load("data/raw/train.npz")
print(data["images"].shape, data["labels"].shape)
