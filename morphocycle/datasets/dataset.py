import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tifffile as tfl
import torchvision.transforms as T
from torchvision import transforms


class CellCycleData(Dataset):
    """
    Dataset class for the single cell dataset
    """

    def __init__(
        self,
        img_dir=None,
        split="train",
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.img_dir = Path(img_dir)
        self.split = split

        # Get all the image files
        img_files = list(self.img_dir.glob("**/*/*/*.tif"))
        self.img_files = [
            i
            for i in img_files
            if ("NotKnown" not in str(i))
            and ("G1-S" not in str(i))
            and ("S-G2" not in str(i))
            and ("G2-M" not in str(i))
        ]

        # Get all the labels
        labels = [str(x.parent.name) for x in self.img_files]
        self.label_dict = {
            "G1": 0,
            # "G1-S": 10,
            "S": 1,
            # "S-G2": 30,
            "G2": 2,
            # "G2-M": 50,
            "M": 3,
        }

        self.labels = [self.label_dict[x] for x in labels]
        self.track_ids = [str(x.parent.parent.name) for x in self.img_files]
        self.slide_ids = [str(x.parent.parent.parent.name) for x in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __return_labels__(self, idx):
        img_path = self.img_files[idx]
        img = tfl.imread(img_path)

        if (img == 0).sum() >= (0.5 * img.shape[0] * img.shape[1]):
            return None
        else:
            label = torch.tensor(self.labels[idx])
            return label

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = tfl.imread(img_path)
        # v_min, v_max = img.min(), img.max()
        # new_min, new_max = 0, 255
        # img = np.clip(img, 0, None)
        # img = img/(img.max() + 1e-5)
        # TODO: check if this is the correct way to do this
        if (img == 0).sum() >= (0.5 * img.shape[0] * img.shape[1]):
            return None
        else:
            label = torch.tensor(self.labels[idx])

            img = (img - img.min()) / (img.max() - img.min() + 1e-5)

            # else:
            #     label=torch.tensor(4)
            if self.split == "train":
                transform = T.Compose(
                    [
                        T.ToTensor(),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomRotation(degrees=90),
                        # T.RandomPerspective(distortion_scale=0.5, p=0.5),
                        T.Resize((64, 64)),
                    ]
                )
            else:
                transform = T.Compose([T.ToTensor(), T.Resize((64, 64))])

            img = transform(img)
            img = img.expand(3, *img.shape[1:]).type(torch.FloatTensor)

            track_id = self.track_ids[idx]
            slide_id = self.slide_ids[idx]

            return img, label, track_id, slide_id


class PhaseToFlour(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),

                # Add any other transformations you need
            ]
        ),
    ):
        """
        Args:
            input_dir (string): Directory with all the input images.
            target_dir (string): Directory with all the target images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform

        self.input_images = list(sorted(self.input_dir.glob("**/*.tif")))
        self.target_images = list(sorted(self.target_dir.glob("**/*.tif")))


    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.input_images[idx]
        img_name = Path(img_path.parents[0].name) / img_path.name

        input_img_path = self.input_dir / img_name
        target_img_path = self.target_dir / img_name

        input_image = tfl.imread(input_img_path)
        target_image = tfl.imread(target_img_path)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)


        sample = {"input": input_image.type(
            torch.FloatTensor
        ), "target": target_image.type(torch.FloatTensor)}

        return sample
