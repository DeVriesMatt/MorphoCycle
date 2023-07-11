import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tifffile as tfl


class CellCycleData(Dataset):
    """
    Dataset class for the single cell dataset
    """

    def __init__(
        self,
        img_dir=None,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.img_dir = Path(img_dir)

        # Get all the image files
        img_files = list(self.img_dir.glob("**/*/*/*.tif"))
        self.img_files = [i for i in img_files if "NotKnown" not in str(i)]

        # Get all the labels
        labels = [str(x.parent.name) for x in self.img_files]
        self.label_dict = {"G1": 0,
                           "G1-S": 1,
                           "S": 2,
                           "S-G2": 3,
                           "G2": 4,
                           "G2-M": 5,
                           "M": 6,
                           "MorG1": 7}
        self.labels = [self.label_dict[x] for x in labels]
        self.track_ids = [str(x.parent.parent.name) for x in self.img_files]
        self.slide_ids = [str(x.parent.parent.parent.name) for x in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = tfl.imread(img_path)
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img).type(torch.FloatTensor)
        label = torch.tensor(self.labels[idx])
        track_id = self.track_ids[idx]
        slide_id = self.slide_ids[idx]

        return img, label, track_id, slide_id

