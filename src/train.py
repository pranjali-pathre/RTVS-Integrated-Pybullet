import torch
from depthnet.mm import ModelManager
from depthnet.model import DenseDepth
from depthnet.dataset import DepthDataset
from utils.utils import load_config
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    config = load_config("config.yaml")
    config.update({
        "batch_size": args.batch_size,
        "n_epochs": args.epochs,
        "gpus": args.gpus,
        "checkpoint": args.checkpoint,
    })
    ds = DepthDataset(config["dataset_folder"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DenseDepth().to(device)
    model_manager = ModelManager(config, model, ds)
    model_manager.train(config["checkpoint"])
    model_manager.test()

if __name__ == "__main__":
    main()
