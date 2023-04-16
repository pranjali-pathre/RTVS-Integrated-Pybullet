from multiprocessing import Pool
import os
import numpy as np
import argparse
from utils.utils import load_config
from utils.sim_utils import get_random_config
from utils.logger import logger


def run_trial(obj_pos, obj_vel, _):
    init_cfg = {
        "obj_pos": obj_pos,
        "obj_vel": obj_vel,
        "motion_type": "linear",
    }
    from sim import simulate
    ret = simulate(init_cfg, False, "rtvs", record=True)
    return ret


def get_config_list(cnt, thresh=0.03):
    config_list = []

    def cfg_to_vector(cfg):
        return np.array([*cfg[0].flatten(), *cfg[1].flatten(), cfg[2]])

    while len(config_list) < cnt:
        i = len(config_list)
        print(i)
        cfg = get_random_config()
        cfg_vec = cfg_to_vector(cfg)
        flag = False
        for j in range(i):
            if np.linalg.norm(cfg_to_vector(config_list[j]) - cfg_vec) < thresh:
                flag = True
                break
        if not flag:
            config_list.append(cfg)

    return config_list


def main_helper_task(arg):
    trial_no, trial_config = arg
    state = run_trial(*trial_config)
    np.savez_compressed(
        os.path.join(folder_name, f"trial_{str(trial_no).zfill(5)}.npz"), **state
    )


def gen_dataset(trials, seed, procs):
    global folder_name
    global config
    config = load_config()
    np.random.seed(seed)
    logger.set_level("critical")
    folder_name = config["raw_dataset_folder"]
    os.makedirs(folder_name, exist_ok=True)

    config_lists = get_config_list(trials)

    pool = Pool(procs)
    pool.map(main_helper_task, enumerate(config_lists))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", "-t", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--procs", "-p", type=int, default=10)
    args = parser.parse_args()
    gen_dataset(args.trials, args.seed, args.procs)


if __name__ == "__main__":
    main()
