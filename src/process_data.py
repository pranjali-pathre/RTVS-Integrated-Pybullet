import argparse
from multiprocessing import Pool
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np
from utils.utils import load_config
from multiprocessing.pool import ThreadPool
import gc


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
        max_ht = max(i[2] for i in state["obj_pos"])
        if max_ht - state["obj_pos"][0][2] > 0.04:
            success += 1
        print(state["obj_pos"][0], state["obj_pos"][-1], max_ht)

    print(f"Success rate: {success / total}, {success}/{total}")


def task_helper(arg):
    data_paths = []

    for i, file_name in arg:
        print(i, file_name)
        state = load_state(os.path.join(rawlog_dir, file_name))
        pool = ThreadPool()
        images = state["images"].item()
        rgbs = np.asarray(images.pop("rgb")[_slice], dtype=np.float32) / 255
        depths = (
            np.asarray(images.pop("depth")[_slice], dtype=np.float32) / 2
        )  # max depth val is 1.9x
        rgbds = np.concatenate((rgbs, np.expand_dims(depths, 3)), axis=3)
        base_names = [f"data_{i}_{j}.npz" for j in range(len(rgbds))]

        def save_data(
            file,
            obj_pos,
            obj_vel,
            obj_corners,
            ee_pos,
            ee_vel,
            joint_vels,
            time,
            action,
            cam_eye,
            rgbd,
            pcd_3d,
            pcd_rgb,
            cam_int,
            cam_ext,
        ):
            np.savez_compressed(
                file,
                obj_pos=obj_pos,
                obj_vel=obj_vel,
                obj_corners=obj_corners,
                ee_pos=ee_pos,
                ee_vel=ee_vel,
                joint_vels=joint_vels,
                time=time,
                action=action,
                cam_eye=cam_eye,
                rgbd=rgbd,
                pcd_3d=pcd_3d,
                pcd_rgb=pcd_rgb,
                cam_int=cam_int,
                cam_ext=cam_ext,
            )

        pool.map(
            lambda args: save_data(*args),
            zip(
                [os.path.join(ds_dir, f) for f in base_names],
                state["obj_pos"][_slice],
                np.gradient(state["obj_pos"][_slice], axis=0)
                .mean(axis=0)
                .reshape(1, 3)
                .repeat(len(state["obj_pos"][_slice]), axis=0),
                state["obj_corners"][_slice].flatten(),
                state["ee_pos"][_slice],
                np.gradient(state["ee_pos"][_slice], axis=0),
                state["joint_vel"][_slice],
                state["t"][_slice],
                state["action"][_slice],
                state["cam_eye"][_slice],
                rgbds,
                state["pcd_3d"][_slice],
                state["pcd_rgb"][_slice],
                state["cam_int"][_slice],
                state["cam_ext"][_slice],
            ),
        )
        pool.close()
        data_paths.append(base_names)
        del state
        gc.collect()
    return data_paths


def process_logs(exp_log_dir, n_procs=10):
    global _slice, rawlog_dir, ds_dir
    _slice = slice(0, None)  # algo_working_time_slice
    rawlog_dir = os.path.join(exp_log_dir, "raw")
    ds_dir = os.path.join(exp_log_dir, "ds")
    shutil.rmtree(ds_dir, ignore_errors=True)
    os.makedirs(ds_dir, exist_ok=True)
    file_names = [f for f in os.listdir(rawlog_dir) if f[:2] != "ds"]
    file_names.sort()
    data_paths = []

    args_list = list(enumerate(file_names))
    chunk_size = int(np.ceil(len(args_list) / n_procs))
    args_list_chunks = []
    print(chunk_size, len(args_list))
    for i in range(n_procs):
        args_list_chunks.append(args_list[i * chunk_size : (i + 1) * chunk_size])

    pool = Pool(n_procs)
    map_ret = pool.map(task_helper, args_list_chunks)
    pool.close()
    for ret in map_ret:
        data_paths += ret
    np.savez_compressed(
        os.path.join(ds_dir, "data_paths.npz"),
        data_paths=data_paths,
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
    process_logs(experiment_logs_dir, args.nprocs)
    # success_rate(rawlog_dir)
