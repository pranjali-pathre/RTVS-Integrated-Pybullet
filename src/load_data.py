import argparse
from multiprocessing import Pool
import os
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
from utils.utils import load_config
from multiprocessing.pool import ThreadPool


def load_state(file_name):
    state = np.load(file_name, allow_pickle=True)
    # state = state[state.files[0]].item()
    return state


def success_rate(rawlog_dir):
    file_names = os.listdir(rawlog_dir)
    total = len(file_names)
    success = 0
    for file_name in file_names:
        state = load_state(os.path.join(rawlog_dir, file_name))
        # print(state["init_config"])
        if state["obj_pos"][-1][2] - state["obj_pos"][0][2] > 0.04:
            success += 1
        print(state["obj_pos"][0], state["obj_pos"][-1])

    print(f"Success rate: {success / total}, {success}/{total}")

def task_helper(arg):
    obj_vels = []
    ee_vels = []
    joint_vels = []
    obj_poses = []
    obj_corners = []
    ee_poses = []
    times = []
    actions = []
    rgbd_paths = []
    cam_eyes = []

    for i, file_name in arg:
        print(i, file_name)
        state = load_state(os.path.join(rawlog_dir, file_name))
        # if state["obj_pos"][-1][2] - state["obj_pos"][0][2] < 0.03:
        #     print("skipping")
        #     continue
        obj_poses.append(state["obj_pos"][_slice])
        obj_corners.append(state["obj_corners"][_slice].flatten())
        ee_poses.append(state["ee_pos"][_slice])
        ee_vels.append(np.gradient(state["ee_pos"][_slice], axis=0))
        obj_vels.append(
            np.gradient(state["obj_pos"][_slice], axis=0).mean(axis=0).reshape(1, 3)
        )
        joint_vels.append(state["joint_vel"][_slice])
        times.append(state["t"][_slice])
        actions.append(state["action"][_slice])
        cam_eyes.append(state["cam_eye"][_slice])

        images = state["images"].item()
        rgbs = images.pop("rgb")[_slice]
        depths = images.pop("depth")[_slice]
        rgbs = np.asarray(rgbs, dtype=np.float32) / 255
        depths = np.asarray(depths, dtype=np.float32) / 2  # max depth val is 1.9x
        rgbds = np.concatenate((rgbs, np.expand_dims(depths, 3)), axis=3)
        pool = ThreadPool()
        rgbd_base_names = [f"{i}_{j}.npz" for j in range(len(rgbds))]
        pool.map(
            lambda arg: np.savez_compressed(arg[0], rgbd=arg[1]),
            zip([os.path.join(ds_dir, f) for f in rgbd_base_names], rgbds),
        )
        pool.close()
        rgbd_paths.append(rgbd_base_names)

    return (
        obj_poses,
        obj_corners,
        ee_poses,
        ee_vels,
        obj_vels,
        joint_vels,
        times,
        actions,
        cam_eyes,
        rgbd_paths
    )

def standardize_logs(exp_log_dir, n_procs=10):
    global _slice, rawlog_dir, ds_dir
    _slice = slice(0, None)  # algo_working_time_slice
    rawlog_dir = os.path.join(exp_log_dir, "raw")
    ds_dir = os.path.join(exp_log_dir, "ds")
    os.makedirs(ds_dir, exist_ok=True)
    file_names = [f for f in os.listdir(rawlog_dir) if f[:2] != "ds"]
    obj_vels = []
    ee_vels = []
    joint_vels = []
    obj_poses = []
    obj_corners = []
    ee_poses = []
    times = []
    actions = []
    rgbd_paths = []
    cam_eyes = []
    file_names.sort()

    args_list = list(enumerate(file_names))
    chunk_size = int(np.ceil(len(args_list)/n_procs))
    args_list_chunks = []
    print(chunk_size, len(args_list))
    for i in range(n_procs):
        args_list_chunks.append(args_list[i*chunk_size:(i+1)*chunk_size])
    
    
    pool = Pool(n_procs)
    map_ret = pool.map(task_helper, args_list_chunks)
    for ret in map_ret:
        obj_poses += ret[0]
        obj_corners += ret[1]
        ee_poses += ret[2]
        ee_vels += ret[3]
        obj_vels += ret[4]
        joint_vels += ret[5]
        times += ret[6]
        actions += ret[7]
        cam_eyes += ret[8]
        rgbd_paths += ret[9]

    def standardize(x):
        x = np.asarray(x)
        x_mean = 0#x.mean(axis=(0, 1))
        x_std = 1#x.std(axis=(0, 1))
        if isinstance(x_std, np.ndarray):
            x_std[x_std == 0] = 1
        return (x - x_mean) / x_std, x_mean, x_std

    obj_poses, obj_poses_mean, obj_poses_std = standardize(obj_poses)
    obj_corners, obj_corners_mean, obj_corners_std = standardize(obj_corners)
    ee_poses, ee_poses_mean, ee_poses_std = standardize(ee_poses)
    ee_vels, ee_vels_mean, ee_vels_std = standardize(ee_vels)
    obj_vels, obj_vels_mean, obj_vels_std = standardize(obj_vels)
    joint_vels, joint_vels_mean, joint_vels_std = standardize(joint_vels)
    times, times_mean, times_std = standardize(times)
    actions, actions_mean, actions_std = standardize(actions)
    cam_eyes, cam_eyes_mean, cam_eyes_std = standardize(cam_eyes)

    np.savez_compressed(
        os.path.join(ds_dir, "combined_logs.npz"),
        obj_poses_mean=obj_poses_mean,
        obj_poses_std=obj_poses_std,
        ee_poses_mean=ee_poses_mean,
        ee_poses_std=ee_poses_std,
        ee_vels_mean=ee_vels_mean,
        ee_vels_std=ee_vels_std,
        obj_vels_mean=obj_vels_mean,
        obj_vels_std=obj_vels_std,
        joint_vels_mean=joint_vels_mean,
        joint_vels_std=joint_vels_std,
        times_mean=times_mean,
        times_std=times_std,
        actions_mean=actions_mean,
        actions_std=actions_std,
        obj_poses=obj_poses,
        ee_poses=ee_poses,
        ee_vels=ee_vels,
        obj_vels=obj_vels,
        joint_vels=joint_vels,
        times=times,
        actions=actions,
        cam_eyes=cam_eyes,
        cam_eyes_mean=cam_eyes_mean,
        cam_eyes_std=cam_eyes_std,
        rgbd_paths=rgbd_paths,
        obj_corners=obj_corners,
        obj_corners_mean=obj_corners_mean,
        obj_corners_std=obj_corners_std,
    )


def visualize_logs_ee_vels(rawlog_dir):
    file_names = os.listdir(rawlog_dir)
    np.random.shuffle(file_names)
    for i, file_name in enumerate(file_names[:100]):
        print(i, file_name)
        state = load_state(os.path.join(rawlog_dir, file_name))
        state.pop("images")
        ee_poses = state["ee_pos"]
        ee_speed = np.linalg.norm(np.gradient(ee_poses, axis=0), axis=1)
        obj_poses = state["obj_pos"]
        obj_vel = np.gradient(obj_poses, axis=0).mean(axis=0).reshape(1, 3)
        plt.plot(ee_speed, label="EE speed")
        plt.ylim(0, 0.065)
        plt.title(f"EE speed with obj vel {obj_vel}")
        plt.show()


if __name__ == "__main__":
    config = load_config()
    experiment_logs_dir = config["experiment_folder"]
    rawlog_dir = config["raw_dataset_folder"]
    parser = argparse.ArgumentParser(description="Standardize logs")
    parser.add_argument("--nprocs", type=int, default=8)
    args = parser.parse_args()

    # preprocess_logs_ee_vels(os.path.join(rawlog_dir, "ee_vels_dataset.npz"))
    # visualize_logs_ee_vels(rawlog_dir)
    standardize_logs(experiment_logs_dir, args.nprocs)
    # success_rate(rawlog_dir)
