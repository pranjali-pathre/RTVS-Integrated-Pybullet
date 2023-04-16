import os
import numpy as np
from utils.utils import load_config
from multiprocessing import Pool
from PIL import Image


def process_state(arg):
    i, file_name = arg
    images = np.load(file_name, allow_pickle=True)["images"].item()
    rgbs = images.pop("rgb")
    depths = images.pop("depth")
    depths = np.asarray(depths) / 2  # max depth val is 1.9x

    rgb_names = []
    depth_names = []

    for j in range(len(rgbs)):
        rgb_names.append(f"{i}_{j}.png")
        depth_names.append(f"{i}_{j}.npz")
        Image.fromarray(rgbs[j]).save(os.path.join(ds_dir, rgb_names[-1]))
        np.savez_compressed(os.path.join(ds_dir, depth_names[-1]), depth=depths[j])

    return list(zip(rgb_names, depth_names))


def standardize_logs(config, procs=8):
    global ds_dir
    ds_dir = config["dataset_folder"]
    os.makedirs(ds_dir, exist_ok=True)
    file_names = os.listdir(config["raw_dataset_folder"])
    file_names.sort()
    file_names = [
        os.path.join(config["raw_dataset_folder"], file_name)
        for file_name in file_names
    ]
    pool = Pool(procs)

    rgb_d_file_names = pool.map(process_state, enumerate(file_names))
    final_pair_names = []
    for i in rgb_d_file_names:
        for j in i:
            final_pair_names.append([j[0], j[1]])

    rgb_d_file_names = np.asarray(final_pair_names, dtype=object).reshape(-1, 2)

    np.savez_compressed(config["dataset_file"], rgb_d_file_names=rgb_d_file_names)


def process_dataset(procs=8):
    config = load_config()
    standardize_logs(config, procs)


if __name__ == "__main__":
    process_dataset()
