from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import to_tensor


class DepthDataset(Dataset):
    def __init__(self, ds_dir: str):
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists()
        file_names_file = self.ds_dir / "dataset.npz"
        self.file_names = np.load(file_names_file, allow_pickle=True)[
            "rgb_d_file_names"
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        rgb_file_name, depth_file_name = self.file_names[idx]
        rgb = Image.open(self.ds_dir / rgb_file_name)
        depth = np.load(self.ds_dir / depth_file_name)["depth"]
        depth = Image.fromarray(depth)
        depth = depth.resize(
            (depth.width // 2, depth.height // 2), Image.Resampling.BILINEAR
        )
        depth = to_tensor(depth)
        rgb = to_tensor(rgb)
        return rgb, depth

    def get_dataloaders(self, batch_size, val_test_split=(0.1, 0.2)):
        val_split, test_split = val_test_split
        assert val_split + test_split < 1
        val_split = int(val_split * len(self))
        test_split = int(test_split * len(self))
        train_split = len(self) - val_split - test_split
        train_ds, val_ds, test_ds = random_split(
            self,
            [train_split, val_split, test_split],
            generator=torch.Generator().manual_seed(42),
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_dl, val_dl, test_dl


def test_dataset():
    ds = DepthDataset("../logs/depth/ds")
    print(len(ds))
    mx = 0
    mn = 1e8
    for rgb, depth in ds:
        print(rgb.shape, depth.max(), depth.min())
        mx = max(mx, depth.max())
        mn = min(mn, depth.min())
    print(mx, mn)


if __name__ == "__main__":
    test_dataset()
