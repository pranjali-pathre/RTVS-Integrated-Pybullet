import numpy as np


def get_random_config(rng=None):
    if rng is None:
        f = np.random.normal
    else:
        f = rng.normal

    grasp_time = 4
    vel = f(loc=[0, 0, 0], scale=[0.08, 0.1, 0], size=3)
    vel[1] = abs(vel[1])
    # grasp_pos = f(loc=[0.6, 0.0, 0.875], scale=[0.15, 0.1, 0], size=3)
    # grasp_pos[2] = -abs(grasp_pos[2])
    # obj_init_pos = grasp_pos - vel * grasp_time
    # obj_init_pos[:2] = np.clip(obj_init_pos[:2], [0.35, -0.07], [0.65, 0.07])
    obj_init_pos = f(loc=[0.45, -0.05, 0.851], scale=[0.15, 0.1, 0], size=3)
    vel[1] = abs(vel[1])
    vel[0] = np.clip(vel[0], -0.05, 0.05)

    return np.round(obj_init_pos, 4), np.round(vel, 4), np.round(grasp_time, 4)


def get_config_list(cnt, thresh=0.03, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    config_list = []

    def cfg_to_vector(cfg):
        return np.array([*cfg[0].flatten(), *cfg[1].flatten(), cfg[2]])

    while len(config_list) < cnt:
        i = len(config_list)
        if verbose:
            print(i, end=" ", flush=True)
        cfg = get_random_config(rng)
        cfg_vec = cfg_to_vector(cfg)
        flag = False
        for j in range(i):
            if np.linalg.norm(cfg_to_vector(config_list[j]) - cfg_vec) < thresh:
                flag = True
                break
        if not flag:
            config_list.append(cfg)
    if verbose:
        print()
    return config_list
