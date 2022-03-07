from pathlib import Path

import numpy as np
import torchvision as V
import tp4.data as data

data_dir = Path(__file__).resolve().parents[1] / "data"


def test_balanced_loader():
    X_train = V.datasets.cifar.CIFAR10(
        root=data_dir.resolve(),
        download=False,
        train=False,
        transform=data.minimal_transform,
    )
    balanced_loader = data.balancedDataLoader(X_train, batch_size=10)
    counts = [0] * 10
    for _, y_batch in balanced_loader:
        for y in y_batch:
            counts[y] += 1
    counts = np.array(counts)
    counts = counts / counts.sum()
    assert all(np.abs(counts - 0.1) < 0.02)
