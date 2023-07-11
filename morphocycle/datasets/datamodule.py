import pytorch_lightning as pl
import torch.utils.data
from .dataset import CellCycleData
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as data


class CellCycleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir=None,
        batch_size=1,
    ):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.img_dir = img_dir


    def setup(self, stage=None):
        train_set = CellCycleData(
            img_dir=self.img_dir,
            )
        # use 20% of training data for validation
        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    def calculate_weights(self):
        dloader = DataLoader(self.train_set, batch_size=1, shuffle=False)
        labels = []
        for d in dloader:
            labels.append(d[1].item())
        labels = np.asarray(labels)
        class_counts = np.bincount(labels)

        # Calculate the inverse of each class frequency
        class_weights = 1.0 / class_counts

        # Now, you can create a weight for each instance in the dataset
        weights = class_weights[labels]
        return torch.from_numpy(weights)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights=self.calculate_weights(), num_samples=len(self.train_set)
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
