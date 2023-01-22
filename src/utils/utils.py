import os
import yaml


def load_config(file_name="config.yaml"):
    with open(file_name, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["experiment_folder"] = os.path.join(
        config["log_dir"], config["experiment_name"]
    )
    config["train_logs_dir"] = os.path.join(config["experiment_folder"], "train_logs")
    config["dataset_log_file"] = os.path.join(
        config["experiment_folder"], "ds", "data_paths.npz"
    )
    config["raw_dataset_folder"] = os.path.join(config["experiment_folder"], "raw")
    return config
