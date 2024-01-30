
import torch.utils.data
from .dataset import CellCycleData, PhaseToFlour
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import lightning as pl
from torchvision import transforms



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
        weight = 1.0 / class_sample_count
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
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            collate_fn=collate_fn,
        )


class PhaseToFlourDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_dir,
        target_dir,
        batch_size=32,
        val_split=0.2,
        num_workers=4,
        transform=None,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        # Transformations can be defined here if not passed in __init__
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229]),
                    # Add any other transformations here
                ]
            )

        # Full dataset
        dataset = PhaseToFlour(self.input_dir, self.target_dir, self.transform)

        # Splitting dataset into train and validation sets
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
