import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import load_config


class BaseDataset(Dataset):
    def __init__(self, logs_file, transform=None):
        self.logs_file = logs_file
        # data paths = N x 125 dim array
        self.data_paths = np.array(
            [
                [os.path.join(os.path.dirname(self.logs_file), i) for i in j]
                for j in np.load(logs_file)["data_paths"]
            ],
            dtype=str,
        )
        self.transform = transform
        self.noise = True

    @staticmethod
    def get_data_point(file_name):
        data = np.load(file_name)
        return {
            "obj_pos": data["obj_pos"],
            "obj_vel": data["obj_vel"],
            "obj_corners": data["obj_corners"],
            "ee_pos": data["ee_pos"],
            "ee_vel": data["ee_vel"],
            "joint_vel": data["joint_vels"],
            "time": data["t"],
            "action": data["action"],
            "cam_eye": data["cam_eye"],
            "rgbd": data["rgbd"],
            "pcd_3d": data["pcd_3d"],
            "pcd_rgb": data["pcd_rgb"],
        }

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class x_rgbd_t_y_rel_obj_conly_Dataset(BaseDataset):
    def __init__(self, logs_file, transform=None):
        super().__init__(logs_file, transform)
        self.data_paths = self.data_paths.flatten()
        self.len = len(self.data_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.get_data_point(self.data_paths[idx])
        rgbd = data["rgbd"]
        t = data["time"]
        obj_pos = data["obj_pos"]
        cam_eye = data["cam_eye"]
        rel_obj_pos = obj_pos - cam_eye
        return (rgbd, t, rel_obj_pos)


DatasetInUse = x_rgbd_t_y_rel_obj_conly_Dataset


def get_dataloaders(ds_cls, config, seed=42, noise=True):
    ds = ds_cls(config["dataset_log_file"])
    if not noise:
        ds.noise = False
    total_size = len(ds)
    test_size = int(total_size * config["test_split"])
    val_size = int(total_size * config["val_split"])
    train_size = total_size - test_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [train_size, val_size, test_size], torch.Generator().manual_seed(seed)
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )
    return train_loader, val_loader, test_loader


def test_ds(ds_cls, logs_file):
    ds = ds_cls(logs_file)
    print(type(ds))
    print("length", len(ds))
    print("ex: ", ds[0][0].shape, ds[0][1].shape)
    for i in range(len(ds)):
        print(i, end="\r", flush=True)
        ds[i]
    print()


def main():
    config = load_config()
    test_ds(DatasetInUse, config["dataset_log_file"])


if __name__ == "__main__":
    main()
