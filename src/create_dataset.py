from depthnet.gen_dataset import gen_dataset
from depthnet.process_dataset import process_dataset
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--trials", "-t", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--procs", "-p", type=int, default=4)
    args = parser.parse_args()
    gen_dataset(args.trials, args.seed, args.procs)
    process_dataset(args.procs)

if __name__ == "__main__":
    main()
