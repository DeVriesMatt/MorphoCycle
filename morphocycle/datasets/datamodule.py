import pytorch_lightning as pl
import torch.utils.data
from .dataset import CellCycleData
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data as data
import torch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

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
        if stage == "train" or stage is None:
            self.train_set = CellCycleData(
                img_dir=self.img_dir,
                )
            # TODO: trying the new data as the validation
            self.valid_set = CellCycleData(
                img_dir="/mnt/nvme0n1/Datasets/PCNA_new/",
                split="val",
                )
        elif stage == "test":
            self.valid_set = CellCycleData(
                img_dir="/mnt/nvme0n1/Datasets/PCNA_new/",
                split="val",
            )
        #     # use 20% of training data for validation
        #     train_set_size = int(len(train_set) * 0.8)
        #     valid_set_size = len(train_set) - train_set_size
        #
        #     # split the train set into two
        #     seed = torch.Generator().manual_seed(42)
        #     self.train_set, self.valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
        # elif stage == "test":
        #     self.valid_set = CellCycleData(
        #         img_dir=self.img_dir,
        #         )


    def calculate_weights(self):
        labels = []
        for i in range(len(self.train_set)):
            if self.train_set[i] is not None:
                labels.append(self.train_set[i][1].item())

        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[labels]
        weights = torch.from_numpy(samples_weight)

        return weights

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights=self.calculate_weights(), num_samples=len(self.train_set)
            ),
            num_workers=24,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=24,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.valid_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=24,
                          collate_fn=collate_fn)
