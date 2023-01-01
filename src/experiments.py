from multiprocessing import Pool
import os
import numpy as np
from sim import URRobotGym
import argparse
from utils.utils import load_config
from utils.sim_utils import get_config_list


def run_trial(obj_init_pos, vel, grasp_time):
    init_config = {
        "obj_init_pos": obj_init_pos,
        "obj_vel": vel,
        "grasp_time": grasp_time,
    }
    env = URRobotGym(
        obj_init_pos, vel, grasp_time, gui=False, inference_mode=inference_mode
    )
    try:
        run_state = env.run()
    except:
        return None
    finally:
        env.pb_client.disconnect()
    run_state["init_config"] = init_config
    return run_state


def main_helper_task(arg):
    trial_no, trial_config = arg
    state = run_trial(*trial_config)
    if state is not None:
        np.savez_compressed(
            os.path.join(folder_name, f"trial_{str(trial_no).zfill(5)}.npz"), **state
        )


def main():
    global folder_name, inference_mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", "-t", type=int, default=1)
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--procs", "-p", type=int, default=10)
    args = parser.parse_args()
    np.random.seed(args.seed)
    folder_name = config["raw_dataset_folder"]
    os.makedirs(config["raw_dataset_folder"], exist_ok=True)
    inference_mode = args.inference

    config_lists = get_config_list(args.trials)
    np.save(os.path.join(config["raw_dataset_folder"], "ds_configs.npy"), np.array(config_lists, dtype=object))

    pool = Pool(args.procs)
    pool.map(main_helper_task, enumerate(config_lists))


if __name__ == "__main__":
    config = load_config()
    main()
