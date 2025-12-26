from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data.mnist_datamodule import MNISTDataModule


class FakeMNIST(Dataset):
    """Fake MNIST dataset for testing without downloading data."""

    def __init__(self, data_dir, train=True, download=False, transform=None):
        # Create fake data: 60000 train samples or 10000 test samples
        num_samples = 60000 if train else 10000
        # Set seed for reproducibility
        np.random.seed(42)
        # Images: 28x28 grayscale, values between 0 and 255 (like MNIST)
        # Convert to uint8 numpy arrays to match MNIST format
        self.data = (np.random.rand(num_samples, 28, 28) * 255).astype(np.uint8)
        # Labels: integers from 0 to 9
        self.targets = np.random.randint(0, 10, size=num_samples, dtype=np.int64)

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    # Mock MNIST to use fake data instead of downloading
    with patch("src.data.mnist_datamodule.MNIST", FakeMNIST):
        dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
        dm.prepare_data()

        assert not dm.data_train and not dm.data_val and not dm.data_test

        dm.setup()
        assert dm.data_train and dm.data_val and dm.data_test
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
        assert num_datapoints == 70_000

        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
